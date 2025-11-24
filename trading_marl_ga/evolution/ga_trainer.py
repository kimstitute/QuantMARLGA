"""
Genetic Algorithm Trainer

GA + MARL 하이브리드 학습 (RACE 방식 참고)
"""

import numpy as np
import copy
from tqdm import tqdm
from multiprocessing import Pool
from agents.multi_agent_system import MultiAgentSystem
from environment.backtest_env import BacktestEnv
from utils.replay_buffer import SharedReplayBuffer
from config import config


class GATrainer:
    """
    Genetic Algorithm Trainer for Multi-Agent Systems
    
    RACE 논문의 핵심:
    1. Agent-level Crossover: 에이전트 단위로 교차
    2. Elitism: 최고 성능 보존
    3. Hybrid Learning: GA로 탐색 + RL로 착취
    """
    
    def __init__(self, population_size=None, n_generations=None, 
                 env_n_days=200, mutation_prob=None, mutation_scale_ratio=None):
        """
        Args:
            population_size (int): Population 크기 (기본: config.POPULATION_SIZE)
            n_generations (int): 총 세대 수 (기본: config.N_GENERATIONS)
            env_n_days (int): 백테스트 기간 (일)
            mutation_prob (float): 각 파라미터가 변이할 확률 (0.0~1.0)
            mutation_scale_ratio (float): 가중치 크기 대비 노이즈 비율
        """
        self.population_size = population_size or config.POPULATION_SIZE
        self.n_generations = n_generations or config.N_GENERATIONS
        self.env_n_days = env_n_days
        self.mutation_prob = mutation_prob or config.MUTATION_PROB
        self.mutation_scale_ratio = mutation_scale_ratio or config.MUTATION_SCALE_RATIO
        
        # Population 초기화
        print(f"\n{'='*60}")
        print(f"GA Trainer 초기화")
        print(f"{'='*60}")
        print(f"Population 크기: {self.population_size}")
        print(f"세대 수: {self.n_generations}")
        print(f"백테스트 기간: {self.env_n_days}일")
        print(f"돌연변이 확률: {self.mutation_prob}")
        print(f"상대적 노이즈 비율: {self.mutation_scale_ratio*100:.1f}%")
        
        # EA Population (n개)
        self.population = [
            MultiAgentSystem(system_id=i) 
            for i in range(self.population_size)
        ]
        print(f"[OK] EA Population 생성: {len(self.population)}개")
        
        # MARL 팀 (1개, 별도)
        self.marl_team = MultiAgentSystem(system_id=-1)  # -1: MARL 팀 표시
        print(f"[OK] MARL 팀 생성: 1개")
        print(f"[OK] 총 팀 수: {len(self.population) + 1}개")
        
        # 백테스트 환경 (재사용)
        self.env = BacktestEnv(n_days=self.env_n_days)
        print(f"[OK] 백테스트 환경 준비 완료")
        
        # Shared Replay Buffer
        self.shared_replay_buffer = SharedReplayBuffer(capacity=config.BUFFER_CAPACITY)
        print(f"[OK] Shared Replay Buffer 생성 (용량: {config.BUFFER_CAPACITY})")
        
        # 환경별 최적화 설정 출력
        print(f"\n{'='*60}")
        print(f"환경 최적화 설정")
        print(f"{'='*60}")
        print(f"실행 환경: {'Colab' if config.ENV['is_colab'] else 'Local'}")
        print(f"GPU: {config.ENV['gpu_name'] if config.ENV['has_gpu'] else 'CPU Only'}")
        if config.ENV['has_gpu']:
            print(f"GPU 메모리: {config.ENV['gpu_memory_gb']:.1f} GB")
        print(f"배치 크기: {config.BATCH_SIZE}")
        print(f"혼합 정밀도 (FP16): {'활성화' if config.USE_AMP else '비활성화'}")
        print(f"병렬 백테스트: {'활성화 (준비중)' if config.USE_PARALLEL_BACKTEST else '비활성화'}")
        print(f"{'='*60}")
        
        # 진화 기록
        self.fitness_history = []  # 세대별 최고/평균/최저 fitness
        self.best_system = None
        self.best_fitness = -np.inf
        
        print(f"\n")
    
    def evaluate_fitness(self, system, verbose=False):
        """
        시스템의 Fitness 평가 (백테스트 실행)
        
        Args:
            system (MultiAgentSystem): 평가할 시스템
            verbose (bool): 상세 출력 여부
        
        Returns:
            float: Fitness (Sharpe Ratio)
        """
        obs = self.env.reset()
        done = False
        step_count = 0
        
        while not done:
            # 행동 선택
            actions = system.act(obs)
            
            # 환경 진행
            next_obs, rewards, done, info = self.env.step(actions)
            step_count += 1
            
            if not done:
                obs = next_obs
        
        # 성과 지표 계산
        metrics = self.env.get_performance_metrics()
        
        # Fitness = Sharpe Ratio (목표 지표)
        fitness = metrics['sharpe_ratio']
        
        if verbose:
            print(f"  Fitness: {fitness:.4f} (Sharpe Ratio)")
            print(f"  총 수익률: {metrics['total_return']*100:.2f}%")
            print(f"  최대 낙폭: {metrics['max_drawdown']*100:.2f}%")
            print(f"  승률: {metrics['win_rate']*100:.2f}%")
        
        return fitness
    
    def evaluate_population(self, verbose=False):
        """
        전체 Population의 Fitness 평가
        
        Args:
            verbose (bool): 진행 상황 출력 여부
        
        Returns:
            list: 각 시스템의 fitness
        """
        fitnesses = []
        
        iterator = tqdm(self.population, desc="Fitness 평가") if verbose else self.population
        
        for i, system in enumerate(iterator):
            fitness = self.evaluate_fitness(system, verbose=False)
            system.fitness = fitness
            fitnesses.append(fitness)
            
            if not verbose:
                # 간단한 진행 상황
                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{len(self.population)}] Fitness: {fitness:.4f}")
        
        return fitnesses
    
    def tournament_selection(self, tournament_size=3, verbose=False):
        """
        Tournament Selection
        
        Args:
            tournament_size (int): 토너먼트 크기
            verbose (bool): 선택 과정 로그 출력
        
        Returns:
            tuple: (선택된 시스템, 선택된 인덱스)
        """
        # 랜덤하게 tournament_size개 선택
        candidate_indices = np.random.choice(len(self.population), size=tournament_size, replace=False)
        candidates = [self.population[i] for i in candidate_indices]
        
        # 가장 높은 fitness를 가진 시스템 반환
        best_in_tournament = max(candidates, key=lambda x: x.fitness)
        best_idx = self.population.index(best_in_tournament)
        
        if verbose:
            candidate_fitnesses = [self.population[i].fitness for i in candidate_indices]
            print(f"      토너먼트: 팀 {list(candidate_indices)} (fitness: {[f'{f:.3f}' for f in candidate_fitnesses]}) -> 선택: 팀 #{best_idx}")
        
        return best_in_tournament, best_idx
    
    def agent_level_crossover(self, parent1, parent2, parent1_idx, parent2_idx, verbose=False):
        """
        Agent-level Crossover (GA-MARL Hybrid 핵심)
        
        부모 2개에서 각 에이전트를 독립적으로 선택하여 자식 생성
        
        Args:
            parent1 (MultiAgentSystem): 부모 1
            parent2 (MultiAgentSystem): 부모 2
            parent1_idx (int): 부모 1 인덱스
            parent2_idx (int): 부모 2 인덱스
            verbose (bool): 교차 과정 로그 출력
        
        Returns:
            MultiAgentSystem: 자식 시스템
        """
        child = MultiAgentSystem()
        crossover_log = []
        
        # 각 에이전트를 독립적으로 선택 (50% 확률)
        if np.random.rand() < 0.5:
            child.value_agent = parent1.value_agent.clone()
            crossover_log.append(f"Value({parent1_idx})")
        else:
            child.value_agent = parent2.value_agent.clone()
            crossover_log.append(f"Value({parent2_idx})")
        
        if np.random.rand() < 0.5:
            child.quality_agent = parent1.quality_agent.clone()
            crossover_log.append(f"Quality({parent1_idx})")
        else:
            child.quality_agent = parent2.quality_agent.clone()
            crossover_log.append(f"Quality({parent2_idx})")
        
        if np.random.rand() < 0.5:
            child.portfolio_agent = parent1.portfolio_agent.clone()
            crossover_log.append(f"Portfolio({parent1_idx})")
        else:
            child.portfolio_agent = parent2.portfolio_agent.clone()
            crossover_log.append(f"Portfolio({parent2_idx})")
        
        if np.random.rand() < 0.5:
            child.hedging_agent = parent1.hedging_agent.clone()
            crossover_log.append(f"Hedging({parent1_idx})")
        else:
            child.hedging_agent = parent2.hedging_agent.clone()
            crossover_log.append(f"Hedging({parent2_idx})")
        
        if verbose:
            print(f"      교차: {' + '.join(crossover_log)}")
        
        return child
    
    def evolve_generation(self, verbose=True):
        """
        1세대 진화
        
        1. Elitism: 상위 10% 보존
        2. Selection: Tournament Selection
        3. Crossover: Agent-level
        4. Mutation: Gaussian Noise
        
        Args:
            verbose (bool): 상세 진화 과정 로그 출력
        
        Returns:
            dict: 세대 통계 (최고/평균/최저 fitness)
        """
        # 현재 세대 Fitness 평가
        fitnesses = [system.fitness for system in self.population]
        
        # Elitism: 상위 ELITE_FRACTION(20%) 보존 (RACE 논문과 동일)
        n_elite = max(1, int(self.population_size * config.ELITE_FRACTION))
        elite_indices = np.argsort(fitnesses)[-n_elite:]
        elites = [self.population[i].clone() for i in elite_indices]
        
        if verbose:
            elite_ratio = n_elite / self.population_size * 100
            print(f"  [Elitism] 상위 {n_elite}개 보존 ({elite_ratio:.0f}%): 팀 {list(elite_indices)}")
            print(f"            Fitness: {[f'{fitnesses[i]:.3f}' for i in elite_indices]}")
        
        # 새로운 Population 생성
        new_population = elites.copy()
        
        # 나머지는 Selection + Crossover + Mutation
        offspring_count = 0
        while len(new_population) < self.population_size:
            offspring_count += 1
            
            if verbose:
                print(f"\n  [자식 #{offspring_count}]")
            
            # Selection
            parent1, p1_idx = self.tournament_selection(verbose=verbose)
            parent2, p2_idx = self.tournament_selection(verbose=verbose)
            
            # Crossover
            child = self.agent_level_crossover(parent1, parent2, p1_idx, p2_idx, verbose=verbose)
            
            # Mutation
            if np.random.rand() < self.mutation_prob:
                child.mutate(mutation_prob=self.mutation_prob, 
                           mutation_scale_ratio=self.mutation_scale_ratio, 
                           verbose=verbose)
            else:
                if verbose:
                    print(f"      변이: 없음 (확률 {self.mutation_prob*100:.0f}%에서 탈락)")
            
            new_population.append(child)
        
        # Population 교체
        self.population = new_population[:self.population_size]
        
        # 통계
        stats = {
            'max_fitness': np.max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'min_fitness': np.min(fitnesses),
            'std_fitness': np.std(fitnesses)
        }
        
        # 최고 시스템 업데이트
        if stats['max_fitness'] > self.best_fitness:
            self.best_fitness = stats['max_fitness']
            best_idx = np.argmax(fitnesses)
            self.best_system = self.population[best_idx].clone()
        
        if verbose:
            print(f"\n  [진화 완료] 총 {offspring_count}개 자식 생성")
        
        return stats
    
    def rollout_and_evaluate(self, include_marl=True, verbose=False):
        """
        Fitness 평가 + 경험 수집을 동시에 수행 (효율 최적화)
        
        한 번의 백테스트로:
        1. Fitness 측정 (Sharpe Ratio)
        2. 경험 수집 (Shared Buffer에 저장)
        
        EA Population(n개) + MARL 팀(1개) = 총 n+1개 팀 모두 rollout
        
        Args:
            include_marl (bool): MARL 팀 포함 여부
            verbose (bool): 상세 출력
        
        Returns:
            int: 수집된 transition 수
        
        Note:
            병렬 백테스트 (Colab 최적화):
            - config.USE_PARALLEL_BACKTEST = True일 때 활성화
            - multiprocessing.Pool로 구현 가능
            - PyTorch 모델 pickle 이슈 주의
        """
        initial_buffer_size = len(self.shared_replay_buffer)
        
        # TODO: 병렬 백테스트 구현 (Colab에서 5-10배 빠름)
        # if config.USE_PARALLEL_BACKTEST:
        #     with Pool(config.PARALLEL_WORKERS) as pool:
        #         results = pool.map(self._rollout_single_system, self.population)
        
        # 1. EA Population rollout + fitness 측정 (n개)
        ea_portfolio_values = []  # 각 팀의 최종 자산
        ea_returns = []  # 각 팀의 수익률
        
        for i, system in enumerate(self.population):
            obs = self.env.reset()
            done = False
            final_portfolio_value = config.INITIAL_CAPITAL
            
            while not done:
                actions = system.act(obs)
                next_obs, rewards, done, info = self.env.step(actions)
                
                # 최종 자산 추적
                final_portfolio_value = info['portfolio_value']
                
                # Shared buffer에 저장
                if not done:
                    transition = {
                        'value_obs': obs['value_obs'],
                        'value_action': actions['value_scores'],
                        'value_reward': rewards['value_reward'],
                        'value_next_obs': next_obs['value_obs'],
                        
                        'quality_obs': obs['quality_obs'],
                        'quality_action': actions['quality_scores'],
                        'quality_reward': rewards['quality_reward'],
                        'quality_next_obs': next_obs['quality_obs'],
                        
                        'portfolio_obs': obs['portfolio_obs'],
                        'portfolio_action': actions['portfolio_weights'],
                        'portfolio_reward': rewards['portfolio_reward'],
                        'portfolio_next_obs': next_obs['portfolio_obs'],
                        
                        'hedging_obs': obs['hedging_obs'],
                        'hedging_action': actions['hedge_ratios'],
                        'hedging_reward': rewards['hedging_reward'],
                        'hedging_next_obs': next_obs['hedging_obs'],
                        
                        'done': done
                    }
                    self.shared_replay_buffer.add(transition)
                    obs = next_obs
            
            # Fitness 측정 (백테스트 종료 직후)
            metrics = self.env.get_performance_metrics()
            system.fitness = metrics['sharpe_ratio']
            
            # 자산 및 수익률 기록
            return_rate = (final_portfolio_value / config.INITIAL_CAPITAL - 1) * 100
            ea_portfolio_values.append(final_portfolio_value)
            ea_returns.append(return_rate)
            
            if verbose:
                print(f"    EA #{i+1:2d}: 자산={final_portfolio_value:>12,.0f}원 | 수익률={return_rate:>6.2f}% | Fitness={system.fitness:>7.4f}")
        
        # 2. MARL 팀도 rollout + fitness 측정 (1개) ⭐ 핵심!
        marl_portfolio_value = None
        marl_return = None
        
        if include_marl and self.marl_team is not None:
            obs = self.env.reset()
            done = False
            final_portfolio_value = config.INITIAL_CAPITAL
            
            while not done:
                actions = self.marl_team.act(obs)
                next_obs, rewards, done, info = self.env.step(actions)
                
                # 최종 자산 추적
                final_portfolio_value = info['portfolio_value']
                
                if not done:
                    transition = {
                        'value_obs': obs['value_obs'],
                        'value_action': actions['value_scores'],
                        'value_reward': rewards['value_reward'],
                        'value_next_obs': next_obs['value_obs'],
                        
                        'quality_obs': obs['quality_obs'],
                        'quality_action': actions['quality_scores'],
                        'quality_reward': rewards['quality_reward'],
                        'quality_next_obs': next_obs['quality_obs'],
                        
                        'portfolio_obs': obs['portfolio_obs'],
                        'portfolio_action': actions['portfolio_weights'],
                        'portfolio_reward': rewards['portfolio_reward'],
                        'portfolio_next_obs': next_obs['portfolio_obs'],
                        
                        'hedging_obs': obs['hedging_obs'],
                        'hedging_action': actions['hedge_ratios'],
                        'hedging_reward': rewards['hedging_reward'],
                        'hedging_next_obs': next_obs['hedging_obs'],
                        
                        'done': done
                    }
                    self.shared_replay_buffer.add(transition)
                    obs = next_obs
            
            # MARL 팀 fitness 측정
            metrics = self.env.get_performance_metrics()
            self.marl_team.fitness = metrics['sharpe_ratio']
            
            # 자산 및 수익률 기록
            marl_return = (final_portfolio_value / config.INITIAL_CAPITAL - 1) * 100
            marl_portfolio_value = final_portfolio_value
            
            if verbose:
                print(f"    MARL   : 자산={final_portfolio_value:>12,.0f}원 | 수익률={marl_return:>6.2f}% | Fitness={self.marl_team.fitness:>7.4f}")
        
        # 3. 전체 요약 출력
        if verbose and len(ea_portfolio_values) > 0:
            print(f"\n    [요약] EA 팀 ({len(ea_portfolio_values)}개)")
            print(f"      자산 - 최고: {max(ea_portfolio_values):>12,.0f}원 | 평균: {np.mean(ea_portfolio_values):>12,.0f}원 | 최저: {min(ea_portfolio_values):>12,.0f}원")
            print(f"      수익률 - 최고: {max(ea_returns):>6.2f}% | 평균: {np.mean(ea_returns):>6.2f}% | 최저: {min(ea_returns):>6.2f}%")
            if marl_portfolio_value is not None:
                print(f"    [요약] MARL 팀")
                print(f"      자산: {marl_portfolio_value:>12,.0f}원 | 수익률: {marl_return:>6.2f}%")
        
        collected = len(self.shared_replay_buffer) - initial_buffer_size
        return collected
    
    def rl_train_from_shared_buffer(self, system, n_updates=50, verbose=False):
        """
        Shared Replay Buffer에서 RL 학습
        
        1개 MARL 팀이 모든 팀(EA + MARL)의 경험으로 학습
        
        Args:
            system (MultiAgentSystem): 학습할 시스템
            n_updates (int): 업데이트 횟수
            verbose (bool): 상세 출력
        
        Returns:
            MultiAgentSystem: 학습된 시스템
        """
        # 최소 MIN_BUFFER_FOR_RL 이상 있어야 학습 시작
        if len(self.shared_replay_buffer) < config.MIN_BUFFER_FOR_RL:
            if verbose:
                print(f"    [Skip] Buffer 부족 ({len(self.shared_replay_buffer)}/{config.MIN_BUFFER_FOR_RL})")
            return system
        
        total_losses = []
        
        for update_idx in range(n_updates):
            # Shared buffer에서 샘플링
            batch = self.shared_replay_buffer.sample(config.BATCH_SIZE)
            
            # 각 에이전트 업데이트
            losses = system.update_all_from_batch(batch)
            
            if losses:
                total_losses.append(losses)
            
            if verbose and (update_idx + 1) % 10 == 0:
                avg_loss = np.mean([l['total'] for l in total_losses[-10:]])
                print(f"    업데이트 {update_idx+1}/{n_updates}: Loss {avg_loss:.4f}")
        
        return system
    
    def train(self, rl_updates=10):
        """
        전체 학습 루프 (GA + MARL Hybrid)
        
        처음부터 GA + RL Hybrid 동시 수행 (RACE 논문 방식 참고)
        
        Args:
            rl_updates (int): 매 세대 RL 업데이트 횟수
        """
        print(f"\n{'='*60}")
        print(f"GA-MARL Hybrid 학습 시작 (처음부터 동시 수행)")
        print(f"{'='*60}")
        print(f"세대 수: {self.n_generations}")
        print(f"Population: {len(self.population)}개 EA 팀 + 1개 MARL 팀")
        print(f"RL 업데이트/세대: {rl_updates}회")
        print(f"{'='*60}\n")
        
        for gen in range(1, self.n_generations + 1):
            print(f"\n{'='*60}")
            print(f"세대 {gen}/{self.n_generations}")
            print(f"{'='*60}")
            
            # Hybrid Learning 순서 (RACE 방식 참고)
            
            # 1. Rollout + Fitness 평가
            print(f"\n[1/4] Rollout + Fitness 평가: EA({len(self.population)}개) + MARL(1개)")
            collected = self.rollout_and_evaluate(include_marl=True, verbose=True)
            print(f"\n  [OK] 수집된 경험: {collected}개 | 총 Buffer 크기: {len(self.shared_replay_buffer)}")
            
            # Fitness 통계
            fitnesses = [s.fitness if s.fitness is not None else -np.inf 
                        for s in self.population]
            print(f"  EA Fitness: 최고={max(fitnesses):.4f}, " +
                  f"평균={np.mean(fitnesses):.4f}, 최저={min(fitnesses):.4f}")
            if self.marl_team and self.marl_team.fitness:
                print(f"  MARL Fitness: {self.marl_team.fitness:.4f}")
            
            # 2. RL: MARL 팀 학습
            print(f"\n[2/4] RL: MARL 팀 학습 (Shared Buffer 활용)")
            
            # MARL 팀을 Shared buffer에서 학습
            if len(self.shared_replay_buffer) >= config.MIN_BUFFER_FOR_RL:
                self.marl_team = self.rl_train_from_shared_buffer(
                    self.marl_team,
                    n_updates=rl_updates,
                    verbose=False
                )
                print(f"  [OK] MARL 팀 학습 완료 ({rl_updates}회 업데이트)")
            else:
                print(f"  [SKIP] Buffer 부족 ({len(self.shared_replay_buffer)}/{config.MIN_BUFFER_FOR_RL})")
            
            # 3. Injection: MARL 팀을 EA Population의 Worst와 교체
            # 중요: 진화 전에 먼저 injection! (진화 후에는 새 팀들의 fitness가 None)
            print(f"\n[3/4] Injection: MARL 팀을 Population에 주입")
            
            # EA Population fitness로 worst 찾기
            fitnesses = [system.fitness if system.fitness is not None else -np.inf 
                        for system in self.population]
            worst_idx = np.argmin(fitnesses)
            
            # MARL 팀 fitness는 이미 rollout에서 계산됨
            marl_fitness = self.marl_team.fitness if self.marl_team.fitness else -np.inf
            
            print(f"  MARL 팀 Fitness: {marl_fitness:.4f} (RL 학습 후)")
            print(f"  Worst 팀 (#{worst_idx}): {fitnesses[worst_idx]:.4f}")
            
            # Worst와 교체 (MARL 복사본을 Population에 주입)
            self.population[worst_idx] = self.marl_team.clone()
            self.population[worst_idx].fitness = marl_fitness
            
            # 중요: MARL 원본(self.marl_team)은 절대 대체하지 않음!
            # 다음 세대에서도 계속 RL 학습을 이어감
            # Population의 best로 대체하는 것은 잘못된 구현!
            
            print(f"  [OK] Injection 완료 (MARL 복사본 → Population[{worst_idx}])")
            print(f"  [INFO] MARL 원본은 보존, 다음 세대 계속 RL 학습")
            
            # 4. 진화 (GA) - Injection 이후에 수행
            print(f"\n[4/4] 진화 (Selection, Crossover, Mutation)")
            # verbose=True로 설정하면 모든 교차/변이 과정 상세 로그 출력
            stats = self.evolve_generation(verbose=True)
            
            # 세대 통계 (Phase 1과 Phase 2 공통)
            fitnesses = [s.fitness if s.fitness is not None else -np.inf for s in self.population]
            valid_fitnesses = [f for f in fitnesses if f != -np.inf]
            
            if len(valid_fitnesses) > 0:
                current_best = max(valid_fitnesses)
                if self.best_fitness is None or current_best > self.best_fitness:
                    self.best_fitness = current_best
                    best_idx = np.argmax(fitnesses)
                    self.best_system = self.population[best_idx].clone()
                
                stats = {
                    'max_fitness': max(valid_fitnesses),
                    'mean_fitness': np.mean(valid_fitnesses),
                    'min_fitness': min(valid_fitnesses),
                    'std_fitness': np.std(valid_fitnesses)
                }
            else:
                stats = {
                    'max_fitness': 0.0,
                    'mean_fitness': 0.0,
                    'min_fitness': 0.0,
                    'std_fitness': 0.0
                }
            
            # 진행 상황 출력
            print(f"\n{'='*60}")
            print(f"세대 {gen} 완료")
            print(f"{'='*60}")
            print(f"  최고 Fitness: {stats['max_fitness']:.4f}")
            print(f"  평균 Fitness: {stats['mean_fitness']:.4f}")
            print(f"  최저 Fitness: {stats['min_fitness']:.4f}")
            print(f"  표준편차:     {stats['std_fitness']:.4f}")
            print(f"  역대 최고:    {self.best_fitness:.4f}")
            print(f"{'='*60}\n")
            
            # 기록
            self.fitness_history.append(stats)
        
        print(f"\n{'='*60}")
        print(f"학습 완료!")
        print(f"{'='*60}")
        print(f"역대 최고 Fitness: {self.best_fitness:.4f}")
        
        # 최종 평가
        if self.best_system:
            print(f"\n[최종 평가] 최고 성능 시스템")
            print(f"{'='*60}")
            fitness = self.evaluate_fitness(self.best_system, verbose=True)
            print(f"{'='*60}\n")
        
        return self.best_system, self.fitness_history
    
    def save_best_system(self, path="checkpoints/best_system"):
        """최고 성능 시스템 저장"""
        if self.best_system:
            self.best_system.save(path)
            print(f"[OK] 최고 시스템 저장: {path}")
        else:
            print(f"[WARNING] 저장할 시스템이 없습니다.")

