import torch


class SmileRule(torch.nn.Module):
    @staticmethod
    def distance(points, a, b):
        delta = points[a] - points[b]
        return (delta.square().sum()).sqrt()

    def forward(self, landmarks_xy):
        points = landmarks_xy
        mouth_width = self.distance(points, 39, 40)
        eye_a_width = self.distance(points, 16, 18)
        eye_b_width = self.distance(points, 17, 19)

        mouth = self.distance(points, 38, 45) / mouth_width.clamp_min(1e-6)
        eye_a = self.distance(points, 20, 22) / eye_a_width.clamp_min(1e-6)
        eye_b = self.distance(points, 21, 23) / eye_b_width.clamp_min(1e-6)

        valid = (mouth_width > 0) & (eye_a_width > 0) & (eye_b_width > 0)
        smile_like = valid & (mouth >= 0.16) & (eye_a >= 0.45) & (eye_b >= 0.45)

        return smile_like.reshape(1), torch.stack((mouth, eye_a, eye_b))


if __name__ == "__main__":
    torch.onnx.export(
        SmileRule().eval(),
        (torch.zeros(46, 2, dtype=torch.float32),),
        "smile_rule.onnx",
        input_names=["landmarks_xy"],
        output_names=["smile_like", "ratios"],
        opset_version=17,
        dynamo=False,
    )
    print("smile_rule.onnx を作成しました")