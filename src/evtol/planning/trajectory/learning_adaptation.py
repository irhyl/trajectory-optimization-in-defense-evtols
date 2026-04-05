"""
Learning-Based Contingency Adaptation

Adapt contingency detection thresholds and response weights from mission history:
- Mission outcome evaluation: success/failure/partial with metrics
- Weight learning: historical performance → optimal anomaly weights
- Threshold adaptation: environment-specific tuning
- Uncertainty quantification: confidence in learned parameters

**Approach**

1. **Mission Recording**: Capture anomaly detections + responses + outcomes
2. **Offline Learning**: Analyze historical missions (daily/weekly batch)
   - ROC curve analysis: threshold vs detection TPR/FPR
   - Weight optimization: minimize false alarms + maximize prevention
   - Confidence intervals: credible regions for parameters

3. **Online Adaptation**: Gradually blend learned parameters into live system
   - Conservative updates: 10% step from learned parameters
   - Revert if performance degrades

**ROC Analysis**

For each anomaly type, compute:
- Detection Rate (True Positive Rate) vs threshold τ
- False Alarm Rate (False Positive Rate) vs threshold τ
- Optimal τ minimizing: w_fa·FPR + w_miss·(1-TPR)

**References**

[1] Ng & Russell (2000): "Algorithms for Inverse Reinforcement Learning"
    ICML, https://ai.stanford.edu/~ang/papers/icml00-irl.pdf

[2] Bradley (1997): "The use of the area under the ROC curve in the evaluation of machine learning"
    Pattern Recognition, https://doi.org/10.1016/S0031-3203(96)00142-2

[3] Fawcett (2006): "An introduction to ROC analysis"
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2005.10.010
"""

from __future__ import annotations
import logging
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MissionOutcome(Enum):
    """Mission completion status."""
    SUCCESS = "success"                   # All objectives completed
    PARTIAL = "partial"                   # Some objectives completed
    FAILURE = "failure"                   # Aborted/crashed
    CONTINGENCY_TRIGGERED = "contingency" # recovery required


@dataclass
class MissionMetrics:
    """Performance metrics for completed mission."""
    mission_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    outcome: MissionOutcome
    
    # Tracking performance
    position_error_mean: float             # meters (RMS)
    position_error_max: float              # meters
    tracking_efficiency: float             # [0, 1] (lower=better, 1.0=perfect)
    
    # Energy performance
    energy_consumed: float                 # Wh
    energy_estimated: float                # Wh (from plan)
    energy_efficiency: float               # actual/estimated [0.5-1.5 typical]
    
    # Safety events
    contingency_events: int                # Count of contingencies triggered
    false_alarms: int                      # False positive detections
    near_misses: int                       # Critical thresholds crossed
    
    # Environmental context
    wind_speed_mean: float                 # m/s
    threat_level_mean: float               # [0, 1]
    temperature_mean: float                # °C
    
    duration: float = 0.0                  # Mission duration (s)
    
    def __post_init__(self):
        if isinstance(self.timestamp_start, (int, float)):
            self.timestamp_start = datetime.fromtimestamp(self.timestamp_start)
        if isinstance(self.timestamp_end, (int, float)):
            self.timestamp_end = datetime.fromtimestamp(self.timestamp_end)
        
        self.duration = (self.timestamp_end - self.timestamp_start).total_seconds()


@dataclass
class AnomalyHistoryPoint:
    """Single anomaly detection event in mission history."""
    timestamp: datetime
    anomaly_type: str                      # TRACKING_ERROR, ENERGY_OVERRUN, etc.
    severity: float                        # [0, 1] computed severity
    threshold_crossed: bool                # Was contingency triggered?
    response_action: str                   # Action taken (if any)
    outcome: str                           # Did it help? (prevented/worsened/neutral)
    environmental_context: Dict[str, float] # Wind, threat, temp, etc.


@dataclass
class LearnedParameters:
    """Learned contingency parameters from mission history."""
    anomaly_weights: Dict[str, float]      # w_i for each anomaly type
    threshold_multipliers: Dict[str, float]  # τ_i scaling vs default
    confidence_scores: Dict[str, float]    # [0, 1] confidence in learned params
    training_samples: int                  # Number of missions used
    timestamp_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'anomaly_weights': self.anomaly_weights,
            'threshold_multipliers': self.threshold_multipliers,
            'confidence_scores': self.confidence_scores,
            'training_samples': self.training_samples,
            'timestamp_updated': self.timestamp_updated.isoformat(),
        }


class MissionRecorder:
    """Records anomaly detections and mission outcomes for learning."""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.anomaly_history: List[AnomalyHistoryPoint] = []
        self.current_mission_id: Optional[str] = None
        self.current_mission_start: Optional[datetime] = None
    
    def start_mission(self, mission_id: str):
        """Mark beginning of mission."""
        self.current_mission_id = mission_id
        self.current_mission_start = datetime.now()
    
    def record_anomaly(
        self,
        anomaly_type: str,
        severity: float,
        threshold_crossed: bool,
        response_action: str,
        environmental_context: Dict[str, float],
    ):
        """Log anomaly detection event."""
        if not self.current_mission_id:
            logger.warning("No active mission for anomaly recording")
            return
        
        point = AnomalyHistoryPoint(
            timestamp=datetime.now(),
            anomaly_type=anomaly_type,
            severity=severity,
            threshold_crossed=threshold_crossed,
            response_action=response_action,
            outcome="pending",
            environmental_context=environmental_context,
        )
        
        self.anomaly_history.append(point)
        
        # Trim if exceeds max size
        if len(self.anomaly_history) > self.max_history_size:
            self.anomaly_history = self.anomaly_history[-self.max_history_size:]
    
    def finalize_mission(
        self,
        metrics: MissionMetrics,
    ):
        """Mark end of mission with performance metrics."""
        if not self.current_mission_id:
            logger.warning("No active mission to finalize")
            return
        
        logger.info(
            f"Mission {self.current_mission_id} completed: "
            f"{metrics.outcome.value} | "
            f"pos_error={metrics.position_error_mean:.2f}m | "
            f"contingencies={metrics.contingency_events}"
        )
        
        # Update anomaly outcomes based on mission result
        # (TODO: More sophisticated attribution analysis)
        for point in self.anomaly_history[-100:]:  # Last 100 events in mission
            point.outcome = (
                "prevented" if metrics.outcome in [MissionOutcome.SUCCESS, MissionOutcome.PARTIAL]
                else "failed"
            )
        
        self.current_mission_id = None
        self.current_mission_start = None
    
    def get_recent_history(self, anomaly_type: Optional[str] = None, days: int = 7) -> List[AnomalyHistoryPoint]:
        """Retrieve recent anomaly history for learning."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        history = [
            point for point in self.anomaly_history
            if point.timestamp >= cutoff_time
        ]
        
        if anomaly_type:
            history = [p for p in history if p.anomaly_type == anomaly_type]
        
        return history


class ROCAnalyzer:
    """Compute ROC curves and optimal thresholds from anomaly history."""
    
    @staticmethod
    def compute_roc_curve(
        detections: np.ndarray,
        ground_truth: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve from detection scores and ground truth.
        
        **ROC Curve Computation**:
        For varying threshold τ ∈ [0, 1]:
        - TPR(τ) = P(score ≥ τ | true event) = TP / (TP + FN)
        - FPR(τ) = P(score ≥ τ | no event) = FP / (FP + TN)
        
        Args:
            detections: Anomaly severity scores [0, 1]
            ground_truth: Binary labels (1=anomaly occurred, 0=normal)
            thresholds: Specific thresholds to evaluate (or auto-generate)
        
        Returns:
            (fpr_array, tpr_array, thresholds_array)
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        tpr_array = []
        fpr_array = []
        
        positives = np.sum(ground_truth == 1)
        negatives = np.sum(ground_truth == 0)
        
        if positives == 0 or negatives == 0:
            logger.warning("ROC: Imbalanced dataset (no positives or negatives)")
            return np.array([0]), np.array([0]), thresholds
        
        for tau in thresholds:
            detections_positive = detections >= tau
            
            tp = np.sum((detections_positive == 1) & (ground_truth == 1))
            fp = np.sum((detections_positive == 1) & (ground_truth == 0))
            fn = np.sum((detections_positive == 0) & (ground_truth == 1))
            tn = np.sum((detections_positive == 0) & (ground_truth == 0))
            
            tpr = tp / positives if positives > 0 else 0.0
            fpr = fp / negatives if negatives > 0 else 0.0
            
            tpr_array.append(tpr)
            fpr_array.append(fpr)
        
        return np.array(fpr_array), np.array(tpr_array), thresholds
    
    @staticmethod
    def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
        """Area Under ROC Curve (higher is better)."""
        return np.trapz(tpr, fpr)
    
    @staticmethod
    def find_optimal_threshold(
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray,
        cost_false_alarm: float = 1.0,
        cost_miss: float = 2.0,
    ) -> Tuple[float, float]:
        """
        Find threshold minimizing weighted error.
        
        Cost = cost_fa·FPR + cost_miss·(1-TPR)
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Threshold values
            cost_false_alarm: Cost of false alarm
            cost_miss: Cost of missing detection
        
        Returns:
            (optimal_threshold, min_cost)
        """
        cost = cost_false_alarm * fpr + cost_miss * (1.0 - tpr)
        optimal_idx = np.argmin(cost)
        
        return thresholds[optimal_idx], cost[optimal_idx]


class ParameterLearner:
    """Learn optimal contingency parameters from mission history."""
    
    def __init__(self, recorder: MissionRecorder):
        self.recorder = recorder
        self.roc_analyzer = ROCAnalyzer()
        self.learned_params: Optional[LearnedParameters] = None
    
    def learn_from_history(
        self,
        days_lookback: int = 7,
        min_samples_per_type: int = 50,
    ) -> LearnedParameters:
        """
        Learn anomaly weights and thresholds from recent history.
        
        **Algorithm**:
        1. Retrieve recent anomaly history
        2. For each anomaly type:
           a. Compute ROC curve (severity vs outcome)
           b. Find optimal threshold (minimizing false alarms + misses)
           c. Compute weight from frequency + importance
        3. Apply credibility weighting (fewer samples → lower confidence)
        
        Args:
            days_lookback: Historical window (days)
            min_samples_per_type: Minimum detections to learn from
        
        Returns:
            LearnedParameters with weights, thresholds, confidence
        """
        history = self.recorder.get_recent_history(days=days_lookback)
        
        if not history:
            logger.warning("No history available for learning")
            return self._default_parameters()
        
        # Group by anomaly type
        by_type = {}
        for point in history:
            if point.anomaly_type not in by_type:
                by_type[point.anomaly_type] = []
            by_type[point.anomaly_type].append(point)
        
        learned_weights = {}
        learned_thresholds = {}
        confidence_scores = {}
        
        for anom_type, points in by_type.items():
            if len(points) < min_samples_per_type:
                logger.warning(f"{anom_type}: only {len(points)} samples, skipping")
                continue
            
            # Extract severity scores and outcomes
            severities = np.array([p.severity for p in points])
            outcomes = np.array([1 if p.outcome == "prevented" else 0 for p in points])
            
            # Compute ROC
            fpr, tpr, thresholds = self.roc_analyzer.compute_roc_curve(severities, outcomes)
            auc = self.roc_analyzer.compute_auc(fpr, tpr)
            
            # Optimal threshold
            tau_opt, cost_opt = self.roc_analyzer.find_optimal_threshold(
                fpr, tpr, thresholds,
                cost_false_alarm=1.0,
                cost_miss=2.0,  # Prefer preventing missed detections
            )
            
            # Weight from frequency + AUC score
            frequency_weight = len(points) / len(history)
            auc_weight = auc if auc > 0.5 else 0.5  # Degraded confidence if AUC < 0.5
            learned_weights[anom_type] = frequency_weight * auc_weight
            
            # Threshold multiplier vs default (typically 0.3)
            learned_thresholds[anom_type] = tau_opt / 0.3  # Relative to default
            
            # Confidence from sample size + AUC
            confidence_scores[anom_type] = min(1.0, auc * np.sqrt(len(points) / min_samples_per_type))
            
            logger.info(
                f"{anom_type}: samples={len(points)}, auc={auc:.3f}, "
                f"tau_opt={tau_opt:.3f}, weight={learned_weights[anom_type]:.3f}, "
                f"confidence={confidence_scores[anom_type]:.3f}"
            )
        
        # Normalize weights
        total_weight = sum(learned_weights.values())
        if total_weight > 0:
            learned_weights = {k: v / total_weight for k, v in learned_weights.items()}
        else:
            learned_weights = {}
        
        self.learned_params = LearnedParameters(
            anomaly_weights=learned_weights,
            threshold_multipliers=learned_thresholds,
            confidence_scores=confidence_scores,
            training_samples=len(history),
        )
        
        return self.learned_params
    
    def _default_parameters(self) -> LearnedParameters:
        """Return conservative defaults when insufficient data."""
        return LearnedParameters(
            anomaly_weights={},
            threshold_multipliers={},
            confidence_scores={},
            training_samples=0,
        )
    
    def apply_learned_parameters(
        self,
        current_weights: Dict[str, float],
        current_thresholds: Dict[str, float],
        blend_factor: float = 0.1,  # Conservative: 10% step
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Blend learned parameters into live system (conservative update).
        
        Args:
            current_weights: Current anomaly weights
            current_thresholds: Current thresholds
            blend_factor: [0, 1] weight of learned parameters
        
        Returns:
            (updated_weights, updated_thresholds)
        """
        if not self.learned_params:
            logger.warning("No learned parameters available")
            return current_weights, current_thresholds
        
        updated_weights = {}
        for anom_type in current_weights:
            if anom_type in self.learned_params.anomaly_weights:
                confidence = self.learned_params.confidence_scores.get(anom_type, 0.0)
                blend = blend_factor * confidence  # Weight blend by confidence
                
                updated_weights[anom_type] = (
                    (1.0 - blend) * current_weights[anom_type] +
                    blend * self.learned_params.anomaly_weights[anom_type]
                )
            else:
                updated_weights[anom_type] = current_weights[anom_type]
        
        updated_thresholds = {}
        for anom_type in current_thresholds:
            if anom_type in self.learned_params.threshold_multipliers:
                confidence = self.learned_params.confidence_scores.get(anom_type, 0.0)
                blend = blend_factor * confidence
                
                tau_multiplier = self.learned_params.threshold_multipliers[anom_type]
                updated_thresholds[anom_type] = (
                    (1.0 - blend) * current_thresholds[anom_type] +
                    blend * (current_thresholds[anom_type] * tau_multiplier)
                )
            else:
                updated_thresholds[anom_type] = current_thresholds[anom_type]
        
        return updated_weights, updated_thresholds
    
    def save_parameters(self, filepath: Path):
        """Persist learned parameters to file."""
        if not self.learned_params:
            logger.warning("No parameters to save")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.learned_params.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved learned parameters to {filepath}")
    
    def load_parameters(self, filepath: Path) -> LearnedParameters:
        """Load learned parameters from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.learned_params = LearnedParameters(
            anomaly_weights=data['anomaly_weights'],
            threshold_multipliers=data['threshold_multipliers'],
            confidence_scores=data['confidence_scores'],
            training_samples=data['training_samples'],
            timestamp_updated=datetime.fromisoformat(data['timestamp_updated']),
        )
        
        logger.info(f"Loaded learned parameters from {filepath}")
        return self.learned_params
