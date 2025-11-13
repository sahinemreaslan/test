"""
Genetic Algorithm Optimizer using DEAP

Optimizes strategy parameters including:
- Timeframe weights
- Indicator parameters
- Entry/exit thresholds
- Risk management parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any
from deap import base, creator, tools, algorithms
import random
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .fitness_functions import FitnessEvaluator

logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """Optimize strategy parameters using genetic algorithm"""

    def __init__(self, config: Dict):
        """
        Initialize genetic optimizer

        Args:
            config: Configuration dictionary with GA parameters
        """
        self.config = config
        self.ga_config = config.get('genetic_algorithm', {})
        self.fitness_evaluator = FitnessEvaluator()

        # GA parameters
        self.population_size = self.ga_config.get('population_size', 100)
        self.generations = self.ga_config.get('generations', 50)
        self.crossover_prob = self.ga_config.get('crossover_prob', 0.7)
        self.mutation_prob = self.ga_config.get('mutation_prob', 0.2)
        self.tournament_size = self.ga_config.get('tournament_size', 3)
        self.elite_size = self.ga_config.get('elite_size', 5)

        # Parameter bounds
        self.param_bounds = self.ga_config.get('parameters', {})

        # Initialize DEAP
        self._setup_deap()

        # Best individual
        self.best_individual = None
        self.best_fitness = float('-inf')

    def _setup_deap(self):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Define gene ranges (parameter space)
        self.gene_ranges = self._define_gene_ranges()

        # Attribute generator
        for i, (param_name, (low, high)) in enumerate(self.gene_ranges.items()):
            self.toolbox.register(f"attr_{i}", random.uniform, low, high)

        # Structure initializers
        attrs = [getattr(self.toolbox, f"attr_{i}") for i in range(len(self.gene_ranges))]
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attrs, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def _define_gene_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Define parameter ranges for optimization

        Returns:
            Dictionary mapping parameter name to (min, max) tuple
        """
        gene_ranges = {}

        # Timeframe weights (one weight per timeframe)
        timeframes = self.config.get('timeframes', {}).get('all', [])
        tf_weight_bounds = self.param_bounds.get('timeframe_weights', {'min': 0.0, 'max': 10.0})

        for tf in timeframes:
            gene_ranges[f'weight_{tf}'] = (tf_weight_bounds['min'], tf_weight_bounds['max'])

        # Indicator weights
        indicators = ['rsi', 'macd', 'bollinger', 'stochastic', 'ema', 'volume', 'heiken_ashi']
        ind_weight_bounds = self.param_bounds.get('indicator_weights', {'min': 0.0, 'max': 5.0})

        for ind in indicators:
            gene_ranges[f'ind_weight_{ind}'] = (ind_weight_bounds['min'], ind_weight_bounds['max'])

        # RSI thresholds
        gene_ranges['rsi_oversold'] = (
            self.param_bounds.get('rsi_oversold', {}).get('min', 20),
            self.param_bounds.get('rsi_oversold', {}).get('max', 35)
        )
        gene_ranges['rsi_overbought'] = (
            self.param_bounds.get('rsi_overbought', {}).get('min', 65),
            self.param_bounds.get('rsi_overbought', {}).get('max', 80)
        )

        # Stop loss and take profit (ATR multipliers)
        gene_ranges['stop_loss_atr'] = (
            self.param_bounds.get('stop_loss_atr', {}).get('min', 1.0),
            self.param_bounds.get('stop_loss_atr', {}).get('max', 3.0)
        )
        gene_ranges['take_profit_atr'] = (
            self.param_bounds.get('take_profit_atr', {}).get('min', 2.0),
            self.param_bounds.get('take_profit_atr', {}).get('max', 6.0)
        )

        # Position sizing
        gene_ranges['position_size'] = (
            self.param_bounds.get('position_size', {}).get('min', 0.01),
            self.param_bounds.get('position_size', {}).get('max', 0.1)
        )

        # ML confidence threshold
        gene_ranges['ml_confidence_threshold'] = (0.5, 0.9)

        # Fractal score threshold
        gene_ranges['fractal_score_threshold'] = (0.3, 0.8)

        return gene_ranges

    def _custom_mutation(self, individual: List[float]) -> Tuple[List[float]]:
        """
        Custom mutation operator with parameter bounds

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        gene_ranges_list = list(self.gene_ranges.values())

        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                # Gaussian mutation with bounds
                low, high = gene_ranges_list[i]
                mutation_strength = (high - low) * 0.1  # 10% of range

                individual[i] += random.gauss(0, mutation_strength)

                # Clip to bounds
                individual[i] = max(low, min(high, individual[i]))

        return (individual,)

    def decode_individual(self, individual: List[float]) -> Dict[str, float]:
        """
        Decode individual genes to parameter dictionary

        Args:
            individual: List of gene values

        Returns:
            Dictionary mapping parameter name to value
        """
        params = {}
        for i, param_name in enumerate(self.gene_ranges.keys()):
            params[param_name] = individual[i]

        return params

    def evaluate_individual(
        self,
        individual: List[float],
        backtest_func: Callable,
        data: Any
    ) -> Tuple[float]:
        """
        Evaluate individual fitness by running backtest

        Args:
            individual: Individual to evaluate
            backtest_func: Function to run backtest
            data: Data for backtesting

        Returns:
            Tuple with fitness score
        """
        try:
            # Decode individual to parameters
            params = self.decode_individual(individual)

            # Run backtest with these parameters
            equity_curve, trades = backtest_func(data, params)

            # Calculate metrics
            metrics = self.fitness_evaluator.calculate_all_metrics(
                equity_curve,
                trades
            )

            # Calculate fitness score
            optimization_metric = self.config.get('backtesting', {}).get(
                'optimization_metric', 'sharpe_ratio'
            )
            min_trades = self.config.get('backtesting', {}).get('min_trades', 30)

            fitness = self.fitness_evaluator.calculate_fitness_score(
                metrics,
                optimization_metric=optimization_metric,
                min_trades=min_trades
            )

            return (fitness,)

        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            return (float('-inf'),)

    def optimize(
        self,
        backtest_func: Callable,
        data: Any,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Run genetic algorithm optimization

        Args:
            backtest_func: Function to run backtest (signature: func(data, params) -> (equity, trades))
            data: Data for backtesting
            verbose: Print progress

        Returns:
            Tuple of (best parameters, best metrics)
        """
        logger.info(f"Starting GA optimization: {self.generations} generations, "
                   f"population {self.population_size}")

        # Register evaluation function
        self.toolbox.register("evaluate", self.evaluate_individual,
                            backtest_func=backtest_func, data=data)

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of fame (best individuals)
        hof = tools.HallOfFame(self.elite_size)

        # Run evolution
        population, logbook = self._evolve_population(
            population, stats, hof, verbose=verbose
        )

        # Get best individual
        best_ind = hof[0]
        self.best_individual = best_ind
        self.best_fitness = best_ind.fitness.values[0]

        best_params = self.decode_individual(best_ind)

        logger.info(f"Optimization complete. Best fitness: {self.best_fitness:.4f}")

        # Calculate final metrics for best individual
        equity_curve, trades = backtest_func(data, best_params)
        best_metrics = self.fitness_evaluator.calculate_all_metrics(equity_curve, trades)

        return best_params, best_metrics

    def _evolve_population(
        self,
        population: List,
        stats: tools.Statistics,
        hof: tools.HallOfFame,
        verbose: bool = True
    ) -> Tuple[List, tools.Logbook]:
        """
        Evolve population using genetic algorithm

        Args:
            population: Initial population
            stats: Statistics object
            hof: Hall of fame
            verbose: Print progress

        Returns:
            Tuple of (final population, logbook)
        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)

        if verbose:
            logger.info(logbook.stream)

        # Evolution loop
        for gen in range(1, self.generations + 1):
            # Select next generation
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Add elite individuals
            offspring.extend(hof.items)

            # Update hall of fame and population
            hof.update(offspring)
            population[:] = offspring

            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            if verbose and gen % 5 == 0:
                logger.info(logbook.stream)

        return population, logbook

    def get_optimization_history(self, logbook: tools.Logbook) -> pd.DataFrame:
        """
        Get optimization history as DataFrame

        Args:
            logbook: DEAP logbook

        Returns:
            DataFrame with generation statistics
        """
        history = pd.DataFrame(logbook)
        return history
