# ================= GPU TROUBLESHOOTING =====================
# If you see 'Device: cpu' but have a GPU, check the following:
# 1. Run 'nvidia-smi' in your terminal. If it fails, install NVIDIA drivers.
# 2. In Python, run:
#    import torch
#    print(torch.cuda.is_available())
#    print(torch.cuda.device_count())
#    print(torch.version.cuda)
#    # If False or 0, install CUDA-enabled PyTorch:
#    # Go to https://pytorch.org/get-started/locally/ and select your CUDA version.
#    # Example for CUDA 12.1:
#    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 3. Make sure you are using the correct Python environment.
# ===========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from hexagon_peg_rl_env import HexagonPegInHoleRL
import torch

class HexagonPegTrainingCallback(BaseCallback):
    """Custom callback for logging hexagon peg task specific metrics"""
    
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.phase_progress = []
        self.last_print_step = 0
        
    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0:
            
            # Get info from environments
            infos = self.locals.get('infos', [])
            
            if infos:
                # Aggregate metrics
                total_episodes = len(infos)
                successes = sum(1 for info in infos if info.get('success', False))
                avg_phase = np.mean([info.get('task_phase', 1) for info in infos])
                avg_distance = np.mean([info.get('distance_to_hole', 1.0) for info in infos])
                avg_reward = np.mean([info.get('total_reward', 0) for info in infos if 'total_reward' in info])
                
                success_rate = successes / total_episodes if total_episodes > 0 else 0.0
                
                # Log to wandb/tensorboard
                self.logger.record('train/success_rate', success_rate)
                self.logger.record('train/avg_task_phase', avg_phase)
                self.logger.record('train/avg_distance_to_hole', avg_distance)
                self.logger.record('train/avg_episode_reward', avg_reward)
                
                # Print to console with phase and reward
                if self.verbose > 0:
                    print(f"\n{'='*70}")
                    print(f"📊 Step {self.n_calls:,} | Training Progress")
                    print(f"{'='*70}")
                    print(f"  Phase: {avg_phase:.1f} | Reward: {avg_reward:.2f}")
                    print(f"  Success Rate: {success_rate:.1%}")
                    print(f"  Distance to Hole: {avg_distance:.3f}m ({avg_distance*1000:.1f}mm)")
                    print(f"{'='*70}\n")
        
        return True

def create_hexagon_peg_env(xml_path="ur5e_with_DIGIT_primitive_hexagon.xml", 
                          render_mode=None, max_episode_steps=1000):
    """Create and wrap the hexagon peg environment"""
    
    def _init():
        env = HexagonPegInHoleRL(xml_path=xml_path, 
                                render_mode=render_mode, 
                                max_episode_steps=max_episode_steps)
        env = Monitor(env)  # For logging
        return env
    
    return _init

def train_hexagon_peg_agent(
    total_timesteps=1000000,
    n_envs=4,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.05,  # Increased for more exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    gae_lambda=0.95,
    gamma=0.99,
    use_wandb=True,
    project_name="hexagon-peg-in-hole",
    save_model=True,
    model_save_path="./models/hexagon_peg_ppo",
    eval_freq=10000,
    eval_episodes=10
):
    """
    Train PPO agent for hexagon peg-in-hole task
    
    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: PPO learning rate
        batch_size: PPO batch size
        n_epochs: PPO update epochs per rollout
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        gae_lambda: GAE lambda parameter
        gamma: Discount factor
        use_wandb: Whether to use Weights & Biases logging
        project_name: W&B project name
        save_model: Whether to save trained model
        model_save_path: Path to save model
        eval_freq: Evaluation frequency (timesteps)
        eval_episodes: Number of episodes for evaluation
    """
    
    print("="*70)
    print("🚀 HEXAGON PEG-IN-HOLE RL TRAINING")
    print("="*70)
    print(f"Training Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  PPO epochs: {n_epochs}")
    print(f"  Clip range: {clip_range}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Discount factor: {gamma}")
    print("="*70)
    
    # Initialize W&B
    if use_wandb:
        run_name = f"hexagon_peg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=project_name,
            name=run_name,
            config={
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "gae_lambda": gae_lambda,
                "gamma": gamma,
                "algorithm": "PPO",
                "task": "hexagon_peg_in_hole"
            }
        )
        print(f"✓ W&B initialized: {run_name}")
    
    # Create training environments
    print(f"\n🏗️  Creating {n_envs} parallel training environments...")
    train_env = make_vec_env(
        create_hexagon_peg_env(max_episode_steps=1000),
        n_envs=n_envs,
    )
    
    # Normalize observations and rewards
    train_env = VecNormalize(train_env, 
                           norm_obs=True, 
                           norm_reward=True, 
                           clip_obs=10.0, 
                           clip_reward=10.0)
    
    print("✓ Training environments created and normalized")
    
    # Create evaluation environment
    print(f"\n📊 Creating evaluation environment...")
    eval_env = make_vec_env(
        create_hexagon_peg_env(max_episode_steps=1000),
        n_envs=1
    )
    eval_env = VecNormalize(eval_env, 
                          norm_obs=True, 
                          norm_reward=False,  # Don't normalize rewards for eval
                          clip_obs=10.0,
                          training=False)  # Important: set training=False for eval
    
    print("✓ Evaluation environment created")
    
    # Device selection: use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create PPO model
    print(f"\n🧠 Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=2048 // n_envs,  # Rollout length per environment
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device
    )
    
    print("✓ PPO model initialized")
    print(f"   Policy architecture: {model.policy}")
    print(f"   Device: {model.device}")
    
    # Create callbacks
    callbacks = []
    
    # Custom hexagon peg callback
    hexagon_callback = HexagonPegTrainingCallback(log_freq=1000)
    callbacks.append(hexagon_callback)
    
    # W&B callback
    if use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=eval_freq,
            model_save_path=model_save_path,  # Fix: always provide model_save_path
            verbose=2
        )
        callbacks.append(wandb_callback)
    
    # Evaluation callback with early stopping
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=150.0,  # Stop if mean reward > 150
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=eval_freq // n_envs,  # Adjust for parallel envs
        n_eval_episodes=eval_episodes,
        best_model_save_path=f"{model_save_path}_best",
        log_path=f"{model_save_path}_logs",
        verbose=1
    )
    callbacks.append(eval_callback)
    
    print(f"✓ Callbacks configured")
    print(f"   Evaluation frequency: every {eval_freq:,} timesteps")
    print(f"   Early stopping threshold: 150.0 reward")
    
    # Start training
    print(f"\n🎯 Starting training...")
    print(f"   Expected training time: ~{total_timesteps//10000:.0f}-{total_timesteps//5000:.0f} minutes")
    print("="*70)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("⚠️  Training interrupted by user")
        print("="*70)
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None, None, None
    
    # Save final model
    if save_model:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(f"{model_save_path}_final")
        # Also save the VecNormalize wrapper
        train_env.save(f"{model_save_path}_final_vecnormalize.pkl")
        print(f"✓ Final model saved to: {model_save_path}_final")
    
    # Final evaluation
    print(f"\n📈 Running final evaluation...")
    final_rewards = []
    final_successes = []
    
    obs = eval_env.reset()
    for eval_ep in range(eval_episodes):
        episode_reward = 0
        episode_success = False
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            
            if info[0].get('success', False):
                episode_success = True
        
        final_rewards.append(episode_reward)
        final_successes.append(episode_success)
        
        obs = eval_env.reset()
    
    final_success_rate = np.mean(final_successes)
    final_mean_reward = np.mean(final_rewards)
    
    print(f"\n📊 FINAL EVALUATION RESULTS:")
    print(f"   Mean reward: {final_mean_reward:.2f}")
    print(f"   Success rate: {final_success_rate:.1%}")
    print(f"   Episodes evaluated: {eval_episodes}")
    
    # Log final results
    if use_wandb:
        wandb.log({
            "final/mean_reward": final_mean_reward,
            "final/success_rate": final_success_rate,
            "final/training_timesteps": total_timesteps
        })
        wandb.finish()
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model, final_mean_reward, final_success_rate

def test_trained_model(model_path, n_episodes=10, render=True):
    """Test a trained model"""
    
    print(f"\n🧪 Testing trained model: {model_path}")
    
    # Load model and normalization
    try:
        model = PPO.load(f"{model_path}")
        
        # Load VecNormalize if it exists
        try:
            env = make_vec_env(create_hexagon_peg_env(render_mode="human" if render else None), n_envs=1)
            env = VecNormalize.load(f"{model_path}_vecnormalize.pkl", env)
            env.training = False  # Don't update stats during testing
            env.norm_reward = False  # Don't normalize rewards during testing
            print("✓ Loaded model with VecNormalize")
        except:
            env = make_vec_env(create_hexagon_peg_env(render_mode="human" if render else None), n_envs=1)
            print("✓ Loaded model without VecNormalize")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test episodes
    rewards = []
    successes = []
    phase_progress = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_success = False
        max_phase = 1
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1
            
            if info[0].get('success', False):
                episode_success = True
            
            max_phase = max(max_phase, info[0].get('task_phase', 1))
            
            # Print progress
            if step % 100 == 0:
                print(f"  Step {step}: Phase {info[0].get('task_phase', 1)}, "
                      f"Distance: {info[0].get('distance_to_hole', 0):.3f}m")
        
        rewards.append(episode_reward)
        successes.append(episode_success)
        phase_progress.append(max_phase)
        
        result = "SUCCESS ✅" if episode_success else f"Phase {max_phase} 📍"
        print(f"  Result: {result}, Reward: {episode_reward:.2f}, Steps: {step}")
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print(f"   Episodes: {n_episodes}")
    print(f"   Success rate: {np.mean(successes):.1%}")
    print(f"   Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Max reward: {np.max(rewards):.2f}")
    print(f"   Average max phase: {np.mean(phase_progress):.1f}")
    
    env.close()

def main():
    """Main training function"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Train or test hexagon peg-in-hole RL agent")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                       help="Mode: train or test")
    parser.add_argument("--model-path", type=str, default="./models/hexagon_peg_ppo_final",
                       help="Model path for testing")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--test-episodes", type=int, default=10,
                       help="Number of episodes for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🚀 Starting training...")
        model, mean_reward, success_rate = train_hexagon_peg_agent(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            use_wandb=not args.no_wandb
        )
        
        if model is not None:
            print(f"\n✅ Training completed!")
            print(f"   Final success rate: {success_rate:.1%}")
            print(f"   Final mean reward: {mean_reward:.2f}")
        
    elif args.mode == "test":
        print("🧪 Starting testing...")
        test_trained_model(args.model_path, n_episodes=args.test_episodes)

if __name__ == "__main__":
    main()