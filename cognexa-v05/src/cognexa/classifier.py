import torch
from torch import nn


class DocumentClassifier(nn.Module):
    """Classify document representations produced by a Transformer."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert token representations into document-level logits."""

        mask = attention_mask.unsqueeze(-1).float()

        masked_hidden_states = hidden_states * mask

        summed = masked_hidden_states.sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1)

        document_representation = summed / counts

        return self.classifier(document_representation)