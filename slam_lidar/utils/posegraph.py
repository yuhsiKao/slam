import gtsam

class PoseGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.isam = gtsam.ISAM2()


    def add_node(self, idx, T):
        if not self.initial.exists(idx):
            self.initial.insert(idx, gtsam.Pose3(T))

    def add_prior(self):
        noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.initial.insert(0, gtsam.Pose3())
        self.graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), noise))


    def add_odom(self, i, j, T, sigma=0.1):
        noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(T), noise))

    def add_loop(self, i, j, T):
        # Tighter base noise than odometry; Cauchy robust kernel rejects outlier loops
        base = gtsam.noiseModel.Isotropic.Sigma(6, 0.05)
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
            base,
        )
        self.graph.add(gtsam.BetweenFactorPose3(i, j, gtsam.Pose3(T), robust))

    def optimize(self):
        self.isam.update(self.graph, self.initial)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        result = self.isam.calculateEstimate()

        opt_poses = {}
        for k in result.keys():
            opt_poses[k] = result.atPose3(k).matrix()

        return opt_poses