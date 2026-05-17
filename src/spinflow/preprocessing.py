"""
Trajectory I/O: load CSVs from heterogeneous datasets into a small standard schema.

Handles common column name variants, optional finite-difference speed recovery,
direction selection (eastbound vs westbound longitudinal coordinate), and
lane filtering for multi-lane motorways.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


def load_trajectory_csv(path: str,
                        col_map: Optional[Dict[str,str]] = None,
                        fps: float = 24.0) -> pd.DataFrame:
    """
    Load a trajectory CSV and normalize columns to: veh, t, x, y, vx, vy, lane.

    If col_map is None, common header names are auto-detected. When longitudinal
    speed components are missing, vx/vy are filled by central differences with
    dt = 1/fps along each vehicle track.
    """
    df = pd.read_csv(path)
    if col_map is None:
        col_map = {}
        for k in ['veh','track','id','vehicle_id','vehicleID','object_id']:
            if k in df.columns: col_map['veh']=k; break
        for k in ['t','time','time(s)','timestamp','frame_time']:
            if k in df.columns: col_map['t']=k; break
        for k in ['x','pos_x','cx','lon_x','longitudinalDistance(m)']:
            if k in df.columns: col_map['x']=k; break
        for k in ['y','pos_y','cy','lat_y','lateralDistance(m)']:
            if k in df.columns: col_map['y']=k; break
        for k in ['vx','vel_x','vx_mps']:
            if k in df.columns: col_map['vx']=k; break
        for k in ['vy','vel_y','vy_mps']:
            if k in df.columns: col_map['vy']=k; break
        for k in ['lane','lane_id','laneID','Lane','laneId']:
            if k in df.columns: col_map['lane']=k; break

    df = df.rename(columns={
        col_map.get('veh','veh'): 'veh',
        col_map.get('t','t'): 't',
        col_map.get('x','x'): 'x',
        col_map.get('y','y'): 'y'
    })
    if 'vx' in col_map and col_map['vx'] in df.columns:
        df = df.rename(columns={col_map['vx']: 'vx'})
    else:
        df['vx'] = np.nan
    if 'vy' in col_map and col_map['vy'] in df.columns:
        df = df.rename(columns={col_map['vy']: 'vy'})
    else:
        df['vy'] = np.nan
    if 'lane' in col_map and col_map['lane'] in df.columns:
        df = df.rename(columns={col_map['lane']: 'lane'})
    else:
        df['lane'] = -1

    df = df.sort_values(['veh','t'])
    if df['vx'].isna().any():
        dt_est = 1.0/fps
        for vid, g in df.groupby('veh'):
            dx = g['x'].diff().fillna(0.0)
            dy = g['y'].diff().fillna(0.0)
            df.loc[g.index, 'vx'] = dx/dt_est
            df.loc[g.index, 'vy'] = dy/dt_est
    return df


def select_direction(df: pd.DataFrame, direction: str, road_len: float) -> pd.DataFrame:
    """
    Keep one driving direction and build monotone chainage s.

    direction 'eb': keep vehicles whose x increases; s = x.
    direction 'wb': keep decreasing x; s = road_len - x so s still increases downstream.
    """
    g = df.groupby('veh').agg(x0=('x','first'), x1=('x','last'))
    g['dir'] = np.sign(g['x1'] - g['x0']).replace(0, 1)
    dir_map = g['dir'].to_dict()
    df['dir'] = df['veh'].map(dir_map)

    if direction.lower() == 'eb':
        df = df[df['dir'] >= 0].copy()
        df['s'] = df['x']
        df['vs'] = df['vx']
    else:
        df = df[df['dir'] <= 0].copy()
        df['s'] = road_len - df['x']
        df['vs'] = -df['vx']
    return df


def filter_lanes(df: pd.DataFrame, lanes: Optional[List[int]]):
    """Restrict to listed lane IDs; lanes=None keeps all lanes."""
    if lanes is None:
        return df
    return df[df["lane"].isin(lanes)].copy()


def parse_lanes_arg(s: str) -> Optional[List[int]]:
    """Parse CLI lane string: 'all' -> None; '6,7,8' -> [6,7,8]."""
    if s is None or s.lower() == "all":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]
