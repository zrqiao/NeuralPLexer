"""Console script for neuralplexer."""
import argparse
import sys

from omegaconf import OmegaConf

from neuralplexer.model.config import attach_task_config, get_base_config
from neuralplexer.model.training import (setup_pl_datamodule, setup_pl_model,
                                         setup_pl_trainer)


def run_training_loop(config, test_only=False):
    model = setup_pl_model(config)
    dm = setup_pl_datamodule(config, model=model)
    trainer = setup_pl_trainer(config, gpus=config.gpus)
    trainer.logger.watch(model)

    if not test_only:
        trainer.fit(model, datamodule=dm)
    # trainer.validate(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--split-csv", type=str)
    parser.add_argument("--lmdb-path", type=str)
    parser.add_argument("--project-name", default=None, type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--ligand-model", default="piformer", type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--training-frac", default=1, type=int)
    parser.add_argument("--learning-rate", default=None, type=float)
    parser.add_argument("--pretrained-model-path", default=None, type=str)
    parser.add_argument("--ligand-model-path", default=None, type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--freeze-protein-encoder", action="store_true")
    parser.add_argument("--freeze-ligand-encoder", action="store_true")
    parser.add_argument("--discard-ligand", action="store_true")
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--latent-model", type=str)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    config = get_base_config()
    config.project_name = args.project_name
    config.run_name = args.run_name
    config.gpus = args.gpus
    config.mol_encoder.model_name = args.ligand_model
    if args.ligand_model == "megamolbart":
        config.mol_encoder.megamolbart = OmegaConf.load(
            config.mol_encoder.megamolbart_config_path
        )
    if args.ligand_model_path is not None:
        config.mol_encoder.from_pretrained = True
        config.mol_encoder.checkpoint_file = args.ligand_model_path
    config.pretrained_path = args.pretrained_model_path
    config = attach_task_config(config, args.task, dataset=args.dataset)
    config.task.split_csv = args.split_csv
    config.task.training_frac = args.training_frac
    config.task.lmdb_path = args.lmdb_path
    config.task.freeze_protein_encoder = args.freeze_protein_encoder
    config.task.freeze_ligand_encoder = args.freeze_ligand_encoder
    if args.learning_rate is not None:
        config.task.init_learning_rate = args.learning_rate
    if args.discard_ligand:
        config.task.ligands = False
    config.task.use_template = args.use_template
    if args.latent_model is not None:
        config.latent_model = args.latent_model

    run_training_loop(config, test_only=args.test_only)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
