
import os
import shutil
import tempfile
from typing import Any
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import optax
from tunix.rl import robust_trainer
from tunix.sft import peft_trainer

# Mock Model
class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(2, 4, rngs=rngs)
        self.linear2 = nnx.Linear(4, 1, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

def simple_loss_fn(model, x, y):
    pred = model(x)
    loss = (pred - y) ** 2
    return jnp.mean(loss), {"aux_val": 1.0}

def main(_):
    # Setup
    rngs = nnx.Rngs(0)
    model = MLP(rngs=rngs)
    optimizer = optax.sgd(learning_rate=0.1)
    
    # Create Dummy Data
    x_normal = jnp.ones((4, 2))
    y_normal = jnp.ones((4, 1))
    
    # Outlier: make y huge so gradient is huge
    x_outlier = jnp.ones((1, 2))
    y_outlier = jnp.array([[1000.0]]) 
    
    x = jnp.concatenate([x_normal, x_outlier], axis=0)
    y = jnp.concatenate([y_normal, y_outlier], axis=0)
    
    inputs = {"x": x, "y": y}
    
    # Config
    tmp_dir = tempfile.mkdtemp()
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=100,
        checkpoint_root_directory=tmp_dir,
    )
    
    print("Initializing RobustTrainer...")
    # Use small threshold to ensure filtering
    trainer = robust_trainer.RobustTrainer(
        model=model,
        optimizer=optimizer,
        training_config=config,
        custom_checkpoint_metadata_fn=lambda: {},
        curation_threshold=0.1 
    )
    trainer.with_loss_fn(simple_loss_fn, has_aux=True)
    trainer.with_gen_model_input_fn(lambda x: x) 
    
    train_step = trainer.create_train_step_fn()
    train_step = nnx.jit(train_step)
    
    print("Running Train Step...")
    loss, aux = train_step(trainer.model, trainer.optimizer, inputs)
    
    print(f"Loss returned: {loss}")
    if isinstance(aux, dict):
        print(f"Skipped Samples: {aux.get('skipped_samples')}")
        print(f"Grad Norm Mean: {aux.get('grad_norm_mean')}")
        
    if loss < 100.0:
        print("SUCCESS: Outlier was filtered out! Loss is small.")
    else:
        print("FAILURE: Outlier was NOT filtered. Loss is huge.")
        sys.exit(1)

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main(None)
