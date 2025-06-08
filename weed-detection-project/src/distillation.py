import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=3):
        super().__init__()
        self.alpha = alpha
        self.temp = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, labels):
        # Resize logits to match label size if needed
        if student_logits.shape[-2:] != labels.shape[-2:]:
            student_logits = nn.functional.interpolate(
                student_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
            )
        if teacher_logits.shape[-2:] != labels.shape[-2:]:
            teacher_logits = nn.functional.interpolate(
                teacher_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False
            )
            
        # Classification loss
        cls_loss = self.ce_loss(student_logits, labels)
        
        # Distillation loss
        soft_teacher = torch.softmax(teacher_logits / self.temp, dim=1)
        soft_student = torch.log_softmax(student_logits / self.temp, dim=1)
        distill_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (self.temp ** 2)
        
        # Combined loss
        return self.alpha * cls_loss + (1 - self.alpha) * distill_loss

def create_teacher_student():
    # Teacher (large model)
    teacher = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b5",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Student (efficient model)
    student = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    return teacher, student