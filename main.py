import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CheckpointModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, y):
        return y + self.mlp(y)

class PartnerNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, y):
        return self.mlp(y)

class PostNet(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.mlp = nn.Linear(dim, num_classes)

    def forward(self, y):
        return self.mlp(y)

class CheckpointController(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, max_steps):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.checkpoint = CheckpointModule(hidden_dim)
        self.partner = PartnerNet(hidden_dim)
        self.post = PostNet(hidden_dim, num_classes)
        self.max_steps = max_steps
        self.num_classes = num_classes

    def forward(self, x, labels=None, train=True, conf_threshold=0.95, alpha=1.0, lambda_cost=0.01):
        y = self.proj(x)  # y0
        batch_size = x.size(0)
        device = x.device

        if not train:
            # Inference: hard halting per sample
            y_ts = []
            p_ts = []
            y_cur = y
            for _ in range(self.max_steps):
                y_cur = self.checkpoint(y_cur)
                y_ts.append(y_cur.unsqueeze(1))  # batch x 1 x dim
                p = self.partner(y_cur)
                p_ts.append(p.unsqueeze(1))  # batch x 1 x 1

            y_ts = torch.cat(y_ts, dim=1)  # batch x max_steps x dim
            p_ts = torch.cat(p_ts, dim=1)  # batch x max_steps x 1

            # Find the first step where p_t > conf_threshold for each sample
            exceeds = p_ts.squeeze(-1) > conf_threshold  # batch x max_steps
            halting_steps = torch.argmax(exceeds.float(), dim=1)  # index of first True
            # If no halt, use last step
            no_halt = ~exceeds.any(dim=1)
            halting_steps[no_halt] = self.max_steps - 1

            # Gather the y at halting step
            final_y = y_ts[torch.arange(batch_size, device=device), halting_steps]
            logits = self.post(final_y)
            return logits

        else:
            # Training: soft halting
            assert labels is not None, "Labels required for training"
            y_ts = []
            p_ts = []
            loss_ts = []
            y_cur = y
            for _ in range(self.max_steps):
                y_cur = self.checkpoint(y_cur)
                y_ts.append(y_cur.unsqueeze(1))
                logits_cur = self.post(y_cur)
                loss_cur = F.cross_entropy(logits_cur, labels, reduction='none')  # batch
                loss_ts.append(loss_cur.unsqueeze(1))
                p = self.partner(y_cur)
                p_ts.append(p.unsqueeze(1))

            y_ts = torch.cat(y_ts, dim=1)  # batch x steps x dim
            p_ts = torch.cat(p_ts, dim=1)  # batch x steps x 1
            loss_ts = torch.cat(loss_ts, dim=1)  # batch x steps

            # Compute soft halting weights g_t
            one_minus_p = 1 - p_ts.squeeze(-1)  # batch x steps
            # Prefix products: prod_{i < t} (1 - p_i)
            prefix_init = torch.cat([torch.ones(batch_size, 1, device=device), one_minus_p[:, :-1]], dim=1)
            prefix_prod = torch.cumprod(prefix_init, dim=1)  # batch x steps

            g = p_ts.squeeze(-1) * prefix_prod  # batch x steps

            # Add remainder to last g
            remainder = prefix_prod[:, -1] * one_minus_p[:, -1]
            g[:, -1] += remainder

            # Final y as weighted sum
            final_y = torch.sum(g.unsqueeze(-1) * y_ts, dim=1)  # batch x dim
            final_logits = self.post(final_y)
            L_task = F.cross_entropy(final_logits, labels)

            # Partner loss with randomized tau
            max_loss = math.log(self.num_classes)
            tau = torch.rand_like(loss_ts) * max_loss
            h_t = (loss_ts < tau).float()
            L_partner = F.binary_cross_entropy(p_ts.squeeze(-1), h_t)

            # Expected steps
            steps = torch.arange(1, self.max_steps + 1, device=device).float()  # 1 to max_steps
            expected_steps = torch.mean(torch.sum(g * steps, dim=1))

            total_loss = L_task + alpha * L_partner + lambda_cost * expected_steps
            return total_loss

# Example usage:
# model = CheckpointController(input_dim=784, hidden_dim=128, num_classes=10, max_steps=5)
# For training: loss = model(inputs, labels=targets, train=True)
# For inference: preds = model(inputs, train=False)