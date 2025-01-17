from pathlib import Path
import os

import typer
from typing_extensions import Annotated
import pickle

app = typer.Typer()
models_path = "./models"

@app.command()
def train(output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """Train the model."""
    model = [1,2,3,4,5]
    out_path = os.path.join(models_path, output_file)
    print(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    app()
