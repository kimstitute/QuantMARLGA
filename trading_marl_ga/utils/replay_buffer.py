"""
멀티 에이전트 시스템을 위한 공유 리플레이 버퍼

4개의 에이전트가 하나의 리플레이 버퍼를 공유하여 데이터 효율성을 극대화합니다.
각 transition은 모든 4개 에이전트의 관측, 행동, 보상을 포함합니다.
"""

import numpy as np
import torch


class SharedReplayBuffer:
    """
    멀티 에이전트 시스템의 모든 에이전트가 공유하는 리플레이 버퍼
    
    다음 형식의 transition을 저장:
    {
        'value_obs': (obs_dim,),
        'value_action': (action_dim,),
        'value_reward': (1,),
        'value_next_obs': (obs_dim,),
        
        'quality_obs': (obs_dim,),
        'quality_action': (action_dim,),
        'quality_reward': (1,),
        'quality_next_obs': (obs_dim,),
        
        ... (portfolio와 hedging도 동일)
        
        'done': (1,)
    }
    """
    
    def __init__(self, capacity):
        """
        Args:
            capacity (int): 리플레이 버퍼의 최대 크기
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, transition):
        """
        버퍼에 transition을 추가
        
        Args:
            transition (dict): 모든 에이전트의 관측, 행동, 보상을 포함하는 딕셔너리
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # 순환 버퍼: 가장 오래된 transition을 덮어씀
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        무작위로 transition 배치를 샘플링
        
        Args:
            batch_size (int): 샘플링할 transition 개수
            
        Returns:
            dict: 각 값이 torch.FloatTensor인 transition 배치
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"버퍼에 샘플이 충분하지 않습니다. "
                           f"현재 {len(self.buffer)}개, 필요 {batch_size}개")
        
        # 중복 없이 무작위 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # 모든 transition 수집
        batch = {}
        keys = self.buffer[0].keys()
        
        for key in keys:
            samples = [self.buffer[i][key] for i in indices]
            batch[key] = torch.FloatTensor(np.array(samples))
        
        return batch
    
    def __len__(self):
        """버퍼의 현재 크기 반환"""
        return len(self.buffer)
    
    def clear(self):
        """버퍼에서 모든 transition 삭제"""
        self.buffer = []
        self.position = 0
    
    def is_ready(self, batch_size):
        """학습에 충분한 샘플이 있는지 확인"""
        return len(self.buffer) >= batch_size

