"""
数据预处理模块 (Data Preprocessing)

功能描述：
    提供统一的轨迹数据加载与清洗接口，屏蔽不同数据集格式差异。
    负责将原始 CSV 数据转换为算法所需的标准 DataFrame 格式，包含 t, s, v 等核心字段。

主要功能：
    1. 统一列名映射 (Column Mapping): 支持多种命名惯例 (如 vehicle_id vs id)。
    2. 速度估计 (Velocity Estimation): 若原始数据缺失速度，自动通过位置差分计算。
    3. 方向筛选 (Direction Filtering): 支持东向/西向 (EB/WB) 筛选及坐标系转换。
    4. 车道过滤 (Lane Filtering): 支持按车道 ID 筛选特定车流。
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


def load_trajectory_csv(path: str,
                        col_map: Optional[Dict[str,str]] = None,
                        fps: float = 24.0) -> pd.DataFrame:
    """
    读取标准轨迹 CSV；可通过 col_map 指定列名映射：
      col_map = {'veh':'id','t':'time','x':'x','y':'y','vx':'vx','vy':'vy','lane':'lane'}
    若无速度列，将按轨迹差分估计（步长=1/fps）。
    """
    df = pd.read_csv(path)
    if col_map is None:
        col_map = {}
        # 车辆ID
        for k in ['veh','track','id','vehicle_id','vehicleID','object_id']:
            if k in df.columns: col_map['veh']=k; break
        # 时间
        for k in ['t','time','time(s)','timestamp','frame_time']:
            if k in df.columns: col_map['t']=k; break
        # 坐标
        for k in ['x','pos_x','cx','lon_x','longitudinalDistance(m)']:
            if k in df.columns: col_map['x']=k; break
        for k in ['y','pos_y','cy','lat_y','lateralDistance(m)']:
            if k in df.columns: col_map['y']=k; break
        # 速度
        for k in ['vx','vel_x','vx_mps']:
            if k in df.columns: col_map['vx']=k; break
        for k in ['vy','vel_y','vy_mps']:
            if k in df.columns: col_map['vy']=k; break
        # 车道
        for k in ['lane','lane_id','laneID','Lane','laneId']:
            if k in df.columns: col_map['lane']=k; break

    # 标准列
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

    # 差分估计速度（若缺失）
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
    选定行车方向：'eb'（向东，x 递增）或 'wb'（向西，x 递减）。
    若选择 wb，则用 s = road_len - x，使"沿行驶方向"坐标单调递增。
    """
    g = df.groupby('veh').agg(x0=('x','first'), x1=('x','last'))
    g['dir'] = np.sign(g['x1'] - g['x0']).replace(0, 1)  # 0 当作正向
    dir_map = g['dir'].to_dict()
    df['dir'] = df['veh'].map(dir_map)

    if direction.lower() == 'eb':
        df = df[df['dir'] >= 0].copy()
        df['s'] = df['x']               # 行驶方向坐标
        df['vs'] = df['vx']
    else:
        df = df[df['dir'] <= 0].copy()
        df['s'] = road_len - df['x']    # 翻转到"向前为正"
        df['vs'] = -df['vx']
    return df


def filter_lanes(df: pd.DataFrame, lanes: Optional[List[int]]):
    """按 lane-id 过滤；lanes=None 表示不过滤。"""
    if lanes is None:
        return df
    return df[df["lane"].isin(lanes)].copy()


def parse_lanes_arg(s: str) -> Optional[List[int]]:
    """'all' -> None；'6,7,8' -> [6,7,8]"""
    if s is None or s.lower() == "all":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]

