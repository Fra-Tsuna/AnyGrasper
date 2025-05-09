import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg: DictConfig):
    from src.run import main as run
    return run(cfg)

if __name__ == "__main__":
    main()