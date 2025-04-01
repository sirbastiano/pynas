import numpy as np


class FitnessEvaluator:
    """Class to compute fitness functions for balancing FPS and MCC."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=0.1, epsilon=0.01, lambda_=5.0, target_fps=120.0, task='classification'):
        """
        Initializes the fitness evaluator with tunable parameters.
        
        Args:
            alpha (float): Weight for FPS in the weighted sum formula.
            beta (float): Weight for MCC in the weighted sum formula.
            gamma (float): Exponential scaling factor for MCC penalty.
            delta (float): Bias term to avoid MCC reaching zero.
            epsilon (float): Small value for logarithmic stability.
            lambda_ (float): Sigmoid steepness in inverse MCC penalty.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.lambda_ = lambda_
        assert target_fps > 0, "Target FPS must be positive."
        self.target_fps = target_fps
        self.task = task
        assert task in ['classification', 'segmentation'], "Task must be either 'classification' or 'segmentation'."
        

    @staticmethod
    def rebound_mcc(metric):
        norm = (metric + 1) / 2  # Normalizing MCC from [-1, 1] to [0, 1] # TODO: fix 
        return norm
    
    @staticmethod
    def rebound_fps(fps, target_fps=120.0):
        fps_ratio = min(fps / target_fps, 1.0)  # Cap at 1.0 to avoid rewarding excessively high FPS
        return fps_ratio


    def weighted_sum_exponential(self, fps, metric):
        """Computes fitness using weighted sum with exponential MCC penalty."""
        fps = self.rebound_fps(fps, self.target_fps)
        if self.task == 'classification':
            # For segmentation, we want to penalize low FPS more heavily
            metric = self.rebound_mcc(metric)

        return self.alpha * fps + self.beta * metric * np.exp(self.gamma * metric)

    def multiplicative_penalty(self, fps, metric):
        """Computes fitness using FPS multiplied by MCC with a bias."""
        fps = self.rebound_fps(fps, self.target_fps)
        if self.task == 'classification':
            metric = self.rebound_mcc(metric)

        return fps * max(0, metric + self.delta)

    def logarithmic_penalty(self, fps, metric):
        """Computes fitness using logarithmic MCC scaling."""
        fps = self.rebound_fps(fps, self.target_fps)
        if self.task == 'classification':
            metric = self.rebound_mcc(metric)

        return fps * np.log(1 + max(0, metric + self.epsilon))

    def inverse_mcc_penalty(self, fps, metric):
        """Computes fitness by scaling FPS using a sigmoid function of MCC."""
        fps = self.rebound_fps(fps, self.target_fps)
        if self.task == 'classification':
            metric = self.rebound_mcc(metric)

        return fps / (1 + np.exp(-self.lambda_ * metric))

    def stepwise_penalty(self, fps, metric):
        """Computes fitness with a stepwise penalty for negative MCC."""
        fps = self.rebound_fps(fps, self.target_fps)
        if self.task == 'classification':
            metric = self.rebound_mcc(metric)

        return fps * (1 + metric) if metric >= 0 else fps * 0.1



if __name__ == "__main__":   

    # Example usage:
    evaluator = FitnessEvaluator()
    fps = 536.47
    mcc = -0.00926

    print("Weighted Sum Exponential:", evaluator.weighted_sum_exponential(fps, mcc))
    print("Multiplicative Penalty:", evaluator.multiplicative_penalty(fps, mcc))
    print("Logarithmic Penalty:", evaluator.logarithmic_penalty(fps, mcc))
    print("Inverse MCC Penalty:", evaluator.inverse_mcc_penalty(fps, mcc))
    print("Stepwise Penalty:", evaluator.stepwise_penalty(fps, mcc))