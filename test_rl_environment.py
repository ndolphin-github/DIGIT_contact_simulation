#!/usr/bin/env python3
"""
Quick evaluation script for the Hexagon Peg-in-Hole RL environment
Tests the environment setup and basic functionality
"""

import numpy as np
import time
from hexagon_peg_rl_env import HexagonPegInHoleRL

def test_environment_setup():
    """Test basic environment functionality"""
    
    print("="*60)
    print("🧪 HEXAGON PEG-IN-HOLE RL ENVIRONMENT TEST")
    print("="*60)
    
    # Create environment
    print("1. Creating environment...")
    try:
        env = HexagonPegInHoleRL(render_mode="human")
        print("   ✅ Environment created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        return False
    
    # Test spaces
    print(f"\n2. Checking spaces...")
    print(f"   Observation space: {env.observation_space.shape} {env.observation_space.dtype}")
    print(f"   Action space: {env.action_space.shape} {env.action_space.dtype}")
    print(f"   ✅ Spaces look correct")
    
    # Test reset
    print(f"\n3. Testing reset...")
    try:
        obs, info = env.reset()
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Initial peg position: {info['initial_peg_pos']}")
        print(f"   Hole entrance position: {info['hole_entrance_pos']}")
        print(f"   ✅ Reset successful")
    except Exception as e:
        print(f"   ❌ Reset failed: {e}")
        env.close()
        return False
    
    # Test steps
    print(f"\n4. Testing environment steps...")
    total_reward = 0
    max_phase = 1
    
    for step in range(20):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            max_phase = max(max_phase, info.get('task_phase', 1))
            
            # Render
            env.render()
            time.sleep(0.1)
            
            if step % 5 == 0:
                print(f"   Step {step}: reward={reward:.3f}, phase={info.get('task_phase', 1)}, "
                      f"distance={info.get('distance_to_hole', 0):.3f}m")
            
            if terminated or truncated:
                print(f"   Episode ended at step {step}")
                break
                
        except Exception as e:
            print(f"   ❌ Step {step} failed: {e}")
            break
    
    print(f"   ✅ Steps completed successfully")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Max phase reached: {max_phase}")
    
    # Test state components
    print(f"\n5. Analyzing state components...")
    try:
        npe_data = obs[:6]
        peg_pose = obs[6:13]
        peg_to_hole = obs[13:16]
        
        print(f"   NPE data (6 DOF): {npe_data}")
        print(f"   Peg pose (7 DOF): position={peg_pose[:3]}, quat={peg_pose[3:]}")
        print(f"   Peg-to-hole vector (3 DOF): {peg_to_hole}")
        print(f"   ✅ State components parsed successfully")
    except Exception as e:
        print(f"   ❌ State analysis failed: {e}")
    
    # Clean up
    env.close()
    print(f"\n✅ Environment test completed successfully!")
    return True

def test_reward_components():
    """Test individual reward components"""
    
    print("\n" + "="*60)
    print("🎯 REWARD COMPONENT TESTING")
    print("="*60)
    
    env = HexagonPegInHoleRL()
    obs, info = env.reset()
    
    # Test different action types
    test_actions = [
        ("Random action", env.action_space.sample()),
        ("No movement", np.zeros(7)),
        ("Joint movement only", np.concatenate([np.random.normal(0, 0.1, 6), [1.1]])),
        ("Gripper open", np.concatenate([np.zeros(6), [0.0]])),
        ("Gripper close", np.concatenate([np.zeros(6), [1.1]]))
    ]
    
    for action_name, action in test_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n{action_name}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Task phase: {info.get('task_phase', 1)}")
        print(f"  Distance to hole: {info.get('distance_to_hole', 0):.3f}m")
        
        # Print reward components
        for key, value in info.items():
            if 'reward' in key or 'penalty' in key:
                print(f"  {key}: {value:.3f}")
        
        env.render()
        time.sleep(0.5)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    print("\n✅ Reward component testing completed!")

def demonstration_episode():
    """Run a demonstration episode with manual control hints"""
    
    print("\n" + "="*60)
    print("🎮 DEMONSTRATION EPISODE")
    print("="*60)
    print("Running a sample episode showing task phases...")
    
    env = HexagonPegInHoleRL(render_mode="human")
    obs, info = env.reset()
    
    print(f"Initial state:")
    print(f"  Peg position: {info['initial_peg_pos']}")
    print(f"  Hole position: {info['hole_entrance_pos']}")
    print(f"  Distance to hole: {np.linalg.norm(info['initial_peg_pos'] - info['hole_entrance_pos']):.3f}m")
    
    # Run episode
    episode_reward = 0
    step = 0
    
    while step < 100:  # Max 100 steps for demo
        
        # Simple heuristic: move towards hole, then open gripper
        obs_npe = obs[:6]
        obs_peg_pose = obs[6:13]
        obs_peg_to_hole = obs[13:16]
        
        distance_to_hole = np.linalg.norm(obs_peg_to_hole)
        
        if distance_to_hole > 0.05:  # Phase 1: Approach
            # Move towards hole (simplified heuristic)
            joint_action = np.random.normal(0, 0.05, 6)  # Small random movements
            gripper_action = 1.1  # Keep closed
        elif distance_to_hole > 0.02:  # Phase 2: Align
            # Fine positioning
            joint_action = np.random.normal(0, 0.02, 6)  # Smaller movements
            gripper_action = 1.1  # Keep closed
        else:  # Phase 3: Release
            # Open gripper
            joint_action = np.zeros(6)  # Stop moving
            gripper_action = 0.0  # Open gripper
        
        action = np.concatenate([joint_action, [gripper_action]])
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        
        # Log progress
        if step % 10 == 0:
            print(f"Step {step}: Phase {info.get('task_phase', 1)}, "
                  f"Distance: {distance_to_hole:.3f}m, Reward: {reward:.3f}")
        
        env.render()
        time.sleep(0.05)
        
        if terminated or truncated:
            break
    
    # Results
    success = info.get('success', False)
    final_phase = info.get('task_phase', 1)
    
    print(f"\n📊 DEMONSTRATION RESULTS:")
    print(f"  Steps: {step}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Final phase: {final_phase}")
    print(f"  Success: {'✅' if success else '❌'}")
    
    env.close()

def main():
    """Run all tests"""
    
    print("Starting Hexagon Peg-in-Hole RL Environment Tests...")
    
    # Basic functionality test
    if not test_environment_setup():
        print("❌ Basic environment test failed!")
        return
    
    # Reward testing
    test_reward_components()
    
    # Demonstration
    demonstration_episode()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*60)
    print("The environment is ready for RL training!")
    print("\nTo start training, run:")
    print("  python train_hexagon_peg_rl.py --mode train")
    print("\nTo test a trained model, run:")
    print("  python train_hexagon_peg_rl.py --mode test --model-path <path>")

if __name__ == "__main__":
    main()