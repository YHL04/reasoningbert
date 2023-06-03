import argparse
import torch

from memory import Memory
from trainer import BERTTrainer
from dataset import create_stocks_data, create_memory_data, create_binary_data, create_bert_data

from bert import BERT, BlockBERT, BlockBERTlucidrains, ReasoningBERT, RecurrentTrainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--state_len", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--burnin", type=int, default=10)
    parser.add_argument("--rollout", type=int, default=20)

    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(0)

    data, target = create_bert_data(max_files=1000)
    # data, target = create_stocks_data()
    # data, target = create_memory_data()
    # data, target = create_binary_data()

    # bert = BERT(
    #     vocab_size=args.vocab_size,
    #     n_layers=4,
    #     d_model=512,
    #     n_head=8,
    #     p=0.1
    # )
    block_bert1 = BlockBERT(
        vocab_size=args.vocab_size,
        n_layers=4,
        d_model=512,
        n_head=8,
        p=0.1,
        state_in=True,
        bert=True
    )
    # block_bert2 = BlockBERT(
    #     vocab_size=args.vocab_size,
    #     n_layers=4,
    #     d_model=512,
    #     n_head=8,
    #     p=0.1,
    #     state_in=False,
    #     bert=True
    # )

    # reasoning_bert = ReasoningBERT(
    #     vocab_size=args.vocab_size,
    #     h_layers=2,
    #     v_layers=2,
    #     d_model=512,
    #     n_head=8,
    #     p=0.1
    # )
    # block_bert3 = BlockBERTlucidrains(
    #     vocab_size=args.vocab_size,
    #     n_layers=4,
    #     d_model=512,
    #     n_head=8,
    #     p=0.1
    # )
    # block_bert4 = BlockBERTlucidrains(
    #     vocab_size=args.vocab_size,
    #     n_layers=4,
    #     d_model=512,
    #     n_head=8,
    #     p=0.1
    # )

    trainer1 = RecurrentTrainer(
        vocab_size=args.vocab_size,
        n_layers=4,
        d_model=512,
        n_head=8,
        p=0.1,
    )
    # trainer2 = RecurrentTrainer(
    #     vocab_size=args.vocab_size,
    #     n_layers=4,
    #     d_model=512,
    #     n_head=8,
    #     p=0.1,
    # )

    memory1 = Memory(
        data=data,
        target=target,
        dim=512,
        statelen=args.state_len
    )
    # memory2 = Memory(
    #     data=data,
    #     target=target,
    #     dim=512,
    #     statelen=args.state_len
    # )

    bert_trainer1 = BERTTrainer(
        bert=block_bert1,
        trainer=trainer1,
        memory=memory1,
        lr=args.lr,
        batch_size=args.batch_size,
        n_accumulate=2,
        statelen=1,
        burnin=args.burnin,
        rollout=args.rollout,
        use_trainer=False
    )
    filename = f"logs/bert"
    log = open(filename, "w")

    for i in range(args.steps):
        loss, trainer_loss = bert_trainer1.train_step()
        # if use_trainer and i >= 400 and i % 50 == 0:
        #     for _ in range(50):
        #         bert_trainer.use_trainer_independently()

        # agg_loss, ground_truth = bert_trainer.get_trainer_acc()
        print(f"{loss}, {trainer_loss}")
        log.write(f"{loss}, {trainer_loss}\n")
        log.flush()

        if i % 100 == 0:
            torch.save(bert_trainer1.bert, "models/bert")
            torch.save(bert_trainer1.trainer, "models/trainer")


    # bert_trainer2 = BERTTrainer(
    #     bert=block_bert2,
    #     trainer=trainer2,
    #     memory=memory2,
    #     lr=args.lr,
    #     batch_size=args.batch_size,
    #     n_accumulate=2,
    #     statelen=1,
    #     burnin=args.burnin,
    #     rollout=args.rollout,
    #     use_trainer=False
    # )
    #
    # filename = f"logs/block_bert"
    # log = open(filename, "w")
    #
    # for i in range(args.steps):
    #     loss, trainer_loss = bert_trainer2.train_step()
    #     # if use_trainer and i >= 400 and i % 50 == 0:
    #     #     for _ in range(50):
    #     #         bert_trainer.use_trainer_independently()
    #
    #     # agg_loss, ground_truth = bert_trainer.get_trainer_acc()
    #     agg_loss, ground_truth = 0, 0
    #     print(f"{loss.item()}, {agg_loss}, {ground_truth}")
    #
    #     log.write(f"{loss}, {agg_loss}, {ground_truth}\n")
    #     log.flush()

"""
    for name, model, trainer, memory, use_trainer in zip(["bert", "block_bert"],
                                                         [block_bert1, block_bert2],
                                                         [trainer1, trainer2],
                                                         [memory1, memory2],
                                                         [False, False]):

        bert_trainer = BERTTrainer(
            bert=model,
            trainer=trainer,
            memory=memory,
            lr=args.lr,
            batch_size=args.batch_size,
            n_accumulate=2,
            statelen=1,
            burnin=args.burnin,
            rollout=args.rollout,
            use_trainer=use_trainer
        )

        # print('memory before ', memory.state[0])
        # itr = iter(bert_trainer.trainer.parameters())
        # next(itr)
        # next(itr)
        # next(itr)
        # print('before ', next(itr))

        filename = f"logs/{name}"
        log = open(filename, "w")

        for i in range(args.steps):
            loss, trainer_loss = bert_trainer.train_step()
            # if use_trainer and i >= 400 and i % 50 == 0:
            #     for _ in range(50):
            #         bert_trainer.use_trainer_independently()

            # agg_loss, ground_truth = bert_trainer.get_trainer_acc()
            agg_loss, ground_truth = 0, 0
            print(f"{loss.item()}, {agg_loss}, {ground_truth}")

            log.write(f"{loss}, {agg_loss}, {ground_truth}\n")
            log.flush()

        # itr = iter(bert_trainer.trainer.parameters())
        # next(itr)
        # next(itr)
        # next(itr)
        # print('after ', next(itr))
        # print('memory after ', memory.state[0])

        log.close()
        torch.cuda.empty_cache()
"""

if __name__ == "__main__":
    main()
