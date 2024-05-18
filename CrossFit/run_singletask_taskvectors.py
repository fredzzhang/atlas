import os
import numpy as np
import torch
import torch.nn.functional as F

from utils import label_smoothed_nll_loss
from transformers import BartTokenizer, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from bart import MyBart
from t5 import MyT5
from transformers.models.t5.configuration_t5 import T5Config
from dataloader.fewshot_gym_singletask import NLPFewshotGymSingleTaskData
from utils import freeze_embeds, trim_batch

from tqdm import tqdm
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)
from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector
from src.composition import WeightedT5


def get_model_and_tokenizer(model_name):
    if "t5" in model_name:
        return MyT5, T5Tokenizer
    elif "bart" in model_name:
        return MyBart, BartTokenizer
    else:
        raise Exception()


def get_task_vectors(args):
    task_vectors = {}
    for dataset in args.tasks:
        for chk in args.checkpoints:
            if dataset in chk:
                checkpoint_path = chk
                break
        model = MyT5.from_pretrained(args.model)
        model = PeftModel.from_pretrained(model, checkpoint_path)
        aux_vector = {}
        for idx, (k, v) in enumerate(model.named_parameters()):
            aux_vector[k] = v
        task_vectors[dataset] = NonLinearTaskVector(vector=aux_vector)
    return task_vectors


def run(args, logger, prefix):
    task_vectors = get_task_vectors(args)
    MyModelClass, MyTokenizerClass = get_model_and_tokenizer(args.model)
    tokenizer = MyTokenizerClass.from_pretrained(args.model)

    train_data = NLPFewshotGymSingleTaskData(
        logger, args, args.train_file, data_type="train", is_training=True
    )
    dev_data = NLPFewshotGymSingleTaskData(
        logger, args, args.dev_file, data_type="dev", is_training=False
    )

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    best_dev_performance = None
    test_performance = None

    best_model_state_dict = None

    if args.do_train:
        model = MyModelClass.from_pretrained(args.model)
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "k",
                "q",
                "v",
            ],
        )
        model = get_peft_model(model, peft_config)
        print(len(model.state_dict()))
        for idx, (k, p) in enumerate(model.named_parameters()):
            print(idx, k)
        for k in task_vectors.keys():
            print(k, len(task_vectors[k].vector.keys()))
        task_vectors = [v for k, v in task_vectors.items()]
        model = WeightedT5(model, task_vectors, blockwise=False)

        # new_state_dict = {}
        # cntd = 0
        # for key in pretrained_weights:
        #     if "lora" in key:
        #         print(key)
        #         cntd += 1
        #         aux = None
        #         for idx, taskname in enumerate(args.tasks):
        #             if idx == 0:
        #                 aux = 0.2 * task_vectors[taskname].vector[key]
        #             else:
        #                 aux += 0.8 * task_vectors[taskname].vector[key]
        #         new_state_dict[key] = aux
        #     else:
        #         new_state_dict[key] = pretrained_weights[key]
        # model.load_state_dict(new_state_dict)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        optimizer = AdamW(
            model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
        best_dev_performance, best_model_state_dict = train(
            args, logger, model, train_data, dev_data, optimizer, scheduler, prefix
        )
    else:
        best_dev_performance = 0.0
    if args.do_predict:
        # if args.do_train:   and best_model_state_dict is not None:

        model = MyModelClass.from_pretrained(args.model)
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            # the task to train for (sequence-to-sequence language modeling in this case)
            task_type=TaskType.SEQ_2_SEQ_LM,
            # the dimension of the low-rank matrices
            r=16,
            # the scaling factor for the low-rank matrices
            lora_alpha=8,
            # the dropout probability of the LoRA layers
            lora_dropout=0.1,
            target_modules=[
                "k",
                "q",
                "v",
                # "o",
            ],
        )
        model = get_peft_model(model, peft_config)
        pretrained_weights = model.state_dict()
        new_state_dict = {}
        cntd = 0
        for key in pretrained_weights:
            if "lora" in key:
                cntd += 1
                aux = None
                for idx, taskname in enumerate(args.tasks):
                    if idx == 0:
                        aux = 0.2 * task_vectors[taskname].vector[key]
                    else:
                        aux += 0.8 * task_vectors[taskname].vector[key]
                new_state_dict[key] = aux
            else:
                new_state_dict[key] = pretrained_weights[key]
        model.load_state_dict(new_state_dict)
        print(cntd)
        logger.info("Loading checkpoint from CPU")

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()

        data_type = "test" if "test" in args.test_file else "dev"
        test_data = NLPFewshotGymSingleTaskData(
            logger, args, args.test_file, data_type=data_type, is_training=False
        )

        test_data.load_dataset(tokenizer)
        test_data.load_dataloader()

        test_performance = inference(
            model, test_data, prefix, save_predictions=True, verbose=True
        )
        logger.info(
            "%s on %s data: %.2f"
            % (test_data.metric, test_data.data_type, test_performance)
        )

    return best_dev_performance, test_performance


def train(args, logger, model, train_data, dev_data, optimizer, scheduler, prefix):
    model.train()
    global_step = 0
    global_batch = 0
    train_losses = []
    best_performance = -1.0
    stop_training = False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(
            train_data.dataloader, desc="Epoch {}".format(epoch), disable=args.quiet
        ):
            global_batch += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]

            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                decoder_input_ids=batch[2],
                decoder_attention_mask=batch[3],
                is_training=True,
            )
            lm_logits = outputs[0]
            # print(lm_logits.shape)
            lprobs = F.log_softmax(lm_logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(
                lprobs,
                batch[2],
                epsilon=0.1,
                ignore_index=0,
            )
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_batch % args.gradient_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

                if global_step % args.eval_period == 0:
                    model.eval()
                    # reconstruct the model and store coeficient to remove the model from the gpu.
                    curr_performance = inference(
                        model if args.n_gpu == 1 else model.module, dev_data, prefix
                    )
                    logger.info(
                        "Step %d Train loss %.2f %s %s on epoch=%d"
                        % (
                            global_step,
                            np.mean(train_losses),
                            dev_data.metric,
                            curr_performance,
                            epoch,
                        )
                    )
                    train_losses = []
                    if best_performance < curr_performance:
                        best_model_state_dict = {
                            k: v.cpu() for (k, v) in model.state_dict().items()
                        }
                        # model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        # torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                        model.save_pretrained(
                            os.path.join(
                                args.output_dir,
                                f"{prefix}_{args.learning_rate}_{args.train_batch_size}_best-model.pt",
                            )
                        )
                        logger.info(
                            "Not saving model with best %s: %s -> %s on epoch=%d, global_step=%d"
                            % (
                                dev_data.metric,
                                best_performance,
                                curr_performance,
                                epoch,
                                global_step,
                            )
                        )
                        best_performance = curr_performance
                        wait_step = 0
                        stop_training = False
                    else:
                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                            break

                    model.train()

            if global_step >= args.total_steps:
                stop_training = True
                break

        if stop_training:
            break

    # model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    # torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))
    return best_performance, best_model_state_dict


def inference(model, dev_data, prefix, save_predictions=False, verbose=False):
    print("HERE apply the coef into the weights to initialize a new model.")
    print(model.state_dict().keys())
    predictions = []
    # bos_token_id = dev_data.tokenizer.bos_token_id
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        pad_token_id = dev_data.tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        outputs = model.generate(
            input_ids=batch[0],
            attention_mask=batch[1],
            num_beams=dev_data.args.num_beams,
            max_length=dev_data.args.max_output_length,
            decoder_start_token_id=model.config.decoder_start_token_id,
            early_stopping=dev_data.gen_early_stop,
        )
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return dev_data.evaluate(predictions, verbose=verbose)

    #  decoder_start_token_id=model.config.decoder_start_token_id,
