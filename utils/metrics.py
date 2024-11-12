esp = 1e-5

class MetricTracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def dice_coeff(input, target):
    num_in_target = input.size(0)
    smooth = 1.
    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)
    intersection = (pred * truth).sum(1)
    loss = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)
    return loss.mean().item()