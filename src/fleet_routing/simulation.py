"""Tick-based passenger arrivals and vehicle event simulation (cells 9, 10, 27).
All network inputs are explicit; importing this module does not run simulations.
"""
from pathlib import Path
from fleet_routing.config import OUTPUT_DIR
import pickle
import json
import numpy as np
from scipy.sparse import issparse

def sample_action(pi, state):
    """Sample an action from a sparse or dense policy"""
    if issparse(pi):
        row = pi.getrow(state)
        actions = row.indices
        probs = row.data
    else:
        row = pi[state]
        actions = np.where(row > 0)[0]
        probs = row[actions]
    if len(actions) == 0:
        return state
    probs = probs / probs.sum()
    return np.random.choice(actions, p=probs)

def sample_destination(state, Q):
    """Sample a passenger destination"""
    probs = Q[state]
    if probs.sum() < 1e-10:
        return state
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)

def run_simulation_real(pi, M, sim_hours=12, warmup_hours=2,
                        tick_duration=20, verbose=True,
                        lambda_input=None, *, network, beta=30):
    """
    Simulation with discrete matching events.
    
    Parameters
    ----------
    pi : policy (sparse or dense)
    M : fleet size
    sim_hours : total simulation hours (including warmup)
    warmup_hours : warmup period (not counted in statistics)
    tick_duration : seconds per tick
    verbose : print progress
    lambda_input : scaled lambda (if None, use network.lambda_vec)
    
    Returns
    -------
    avg_reward, total_matched, trajectories, stats
    
    trajectories: list of (car_id, link_id, matched_bool, time_sec)
    """
    if M <= 0 or not isinstance(M, (int, np.integer)):
        raise ValueError("M must be a positive integer")
    if not 0 <= warmup_hours < sim_hours:
        raise ValueError("Require 0 <= warmup_hours < sim_hours")
    if tick_duration <= 0 or not np.isclose(3600 / tick_duration, round(3600 / tick_duration)):
        raise ValueError("tick_duration must divide 3600 seconds exactly")
    tau, c, v, Q = network.tau, network.c, network.v, network.Q
    n_links, init_dist = network.n_links, network.init_dist
    lam = network.lambda_vec if lambda_input is None else np.asarray(lambda_input)
    if lam.shape != (n_links,) or not np.all(np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError("lambda_input must be finite, nonnegative and length n_links")
    if np.any(tau <= 0):
        raise ValueError("All tau values must be positive to avoid zero-time loops")
    
    # Convert tau and c from hours to seconds
    tau_sec = tau * 3600
    c_sec = c * 3600
    
    ticks_per_hour = int(3600 / tick_duration)
    n_ticks = int(sim_hours * ticks_per_hour)
    warmup_ticks = int(warmup_hours * ticks_per_hour)
    effective_hours = sim_hours - warmup_hours
    
    # Initialize the fleet
    fleet = []
    for i in range(M):
        loc = np.random.choice(n_links, p=init_dist)
        fleet.append({
            'status': 'empty',
            'location': loc,
            'time_left': tau_sec[loc],
            'destination': None,
        })
    
    # Number of waiting passengers on each link
    waiting_passengers = np.zeros(n_links)
    
    # Statistics
    total_reward = 0.0
    total_matched = 0
    total_unmatched = 0
    trajectories = []
    z_sum = np.zeros(n_links)
    z_count = 0
    
    if verbose:
        print(f'Simulation: M={M}, sim={sim_hours}h, warmup={warmup_hours}h, tick={tick_duration}s')
    
    for tick in range(n_ticks):
        recording = (tick >= warmup_ticks)
        current_time = tick * tick_duration  # seconds
        
        # === 1. Passenger arrivals (Poisson) ===
        expected_arrivals = lam * (tick_duration / 3600)
        arrivals = np.random.poisson(expected_arrivals)
        waiting_passengers += arrivals
        
        # === 2. Vehicle movement and matching ===
        for car_idx, car in enumerate(fleet):
            car['time_left'] -= tick_duration
            
            while car['time_left'] <= 0:
                overshoot = -car['time_left']
                
                if car['status'] == 'empty':
                    s = car['location']
                    
                    if waiting_passengers[s] > 0:
                        # === Matched ===
                        waiting_passengers[s] -= 1
                        car['status'] = 'occupied'
                        dest = sample_destination(s, Q)
                        car['destination'] = dest
                        car['location'] = dest
                        trip_time = c_sec[s, dest] if c_sec[s, dest] > 0 else tau_sec[s]
                        car['time_left'] = trip_time - overshoot
                        if recording:
                            total_reward += v[s, dest] - beta * (tau[s] + c[s, dest])
                            total_matched += 1
                            trajectories.append((car_idx, s, True, current_time))
                    else:
                        # === Not matched ===
                        next_link = sample_action(pi, s)
                        car['location'] = next_link
                        car['time_left'] = tau_sec[next_link] - overshoot
                        if recording:
                            total_reward -= beta * tau[s]
                            total_unmatched += 1
                            trajectories.append((car_idx, s, False, current_time))
                else:
                    # === Occupied trip completed -> vacant ===
                    car['status'] = 'empty'
                    next_link = sample_action(pi, car['location'])
                    car['location'] = next_link
                    car['time_left'] = tau_sec[next_link] - overshoot
                    car['destination'] = None
        
        # === 3. Record the vacant-vehicle distribution ===
        if recording:
            empty_count = np.zeros(n_links)
            for car in fleet:
                if car['status'] == 'empty':
                    empty_count[car['location']] += 1
            z_sum += empty_count
            z_count += 1
        
        # === 4. Print progress ===
        if verbose and (tick + 1) % ticks_per_hour == 0:
            n_empty = sum(1 for car in fleet if car['status'] == 'empty')
            hour = (tick + 1) / ticks_per_hour
            avg_so_far = total_reward / max(hour - warmup_hours, 1e-10) / M
            print(f'  Hour {hour:.0f}: empty={n_empty}/{M} ({100*n_empty/M:.1f}%), '
                  f'matched={total_matched}, waiting={waiting_passengers.sum():.0f}, '
                  f'avg_R={avg_so_far:.2f}')
    
    avg_reward = total_reward / effective_hours / M
    match_rate = total_matched / max(total_matched + total_unmatched, 1)
    
    stats = {
        'total_matched': total_matched,
        'total_unmatched': total_unmatched,
        'match_rate': match_rate,
        'avg_reward': avg_reward,
        'effective_hours': effective_hours,
        'remaining_passengers': waiting_passengers.sum(),
        'z_avg': z_sum / z_count if z_count > 0 else np.ones(n_links),
    }
    
    if verbose:
        print(f'\n=== Results ===')
        print(f'  Avg reward per vehicle per hour: {avg_reward:.2f}')
        print(f'  Total matched: {total_matched}')
        print(f'  Match rate: {match_rate:.4f}')
    
    return avg_reward, total_matched, trajectories, stats

def run_repeated_simulation(pi, name, n_runs=10, M=5000, sim_hours=12,
                             warmup_hours=2, lambda_input=None,
                             save_trajectories=True, *, network, beta=30,
                             output_dir=None, seed=None, tick_duration=20):
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if seed is not None:
        np.random.seed(seed)
    avg_R_list = []
    matched_list = []
    unmatched_list = []
    match_rate_list = []
    remaining_list = []
    z_avg_list = []
    all_traj_runs = []

    for run in range(n_runs):
        print(f'\n{"="*60}')
        print(f'{name} run {run+1}/{n_runs}')
        print(f'{"="*60}')

        avg_R, matched, traj, stats = run_simulation_real(
            pi, M=M, sim_hours=sim_hours, warmup_hours=warmup_hours,
            verbose=False, lambda_input=lambda_input, network=network, beta=beta,
            tick_duration=tick_duration
        )

        avg_R_list.append(stats['avg_reward'])
        matched_list.append(stats['total_matched'])
        unmatched_list.append(stats['total_unmatched'])
        match_rate_list.append(stats['match_rate'])
        remaining_list.append(stats['remaining_passengers'])
        z_avg_list.append(stats['z_avg'])
        if save_trajectories:
            all_traj_runs.append(traj)

        print(f'  avg_R={stats["avg_reward"]:.2f}, match_rate={stats["match_rate"]:.4f}')

    summary = {
        'name': name,
        'avg_R_mean': np.mean(avg_R_list),
        'avg_R_std': np.std(avg_R_list),
        'avg_R_all': np.array(avg_R_list),
        'matched_mean': np.mean(matched_list),
        'matched_std': np.std(matched_list),
        'unmatched_mean': np.mean(unmatched_list),
        'unmatched_std': np.std(unmatched_list),
        'match_rate_mean': np.mean(match_rate_list),
        'match_rate_std': np.std(match_rate_list),
        'remaining_mean': np.mean(remaining_list),
        'remaining_std': np.std(remaining_list),
        'z_avg_mean': np.mean(z_avg_list, axis=0),
        'z_avg_std': np.std(z_avg_list, axis=0),
        'n_runs': n_runs,
    }

    print(f'\n{"="*60}')
    print(f'{name} Summary (n={n_runs})')
    print(f'{"="*60}')
    print(f'avg_R      = {summary["avg_R_mean"]:.2f} ± {summary["avg_R_std"]:.2f}')
    print(f'match_rate = {summary["match_rate_mean"]:.4f} ± {summary["match_rate_std"]:.4f}')
    print(f'matched    = {summary["matched_mean"]:.0f} ± {summary["matched_std"]:.0f}')

    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f'{name}_repeated_M{M}.pkl' 
    with open(save_path, 'wb') as f:
        pickle.dump({
            'summary': summary,
            'parameters': dict(M=M, sim_hours=sim_hours, warmup_hours=warmup_hours,
                               tick_duration=tick_duration, beta=beta, seed=seed,
                               lambda_input=network.lambda_vec if lambda_input is None else lambda_input,
                               std_ddof=0),
            'all_trajectories': all_traj_runs,  # Set save_trajectories=False to omit trajectories from saved results
        }, f)
    json_path = save_path.with_suffix('.summary.json')
    with json_path.open('w') as handle:
        json.dump(summary, handle, indent=2,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
    print(f'\nSaved to {save_path}')
    print(f'Summary saved to {json_path}')

    return summary
