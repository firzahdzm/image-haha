#!/usr/bin/env python3
"""
Best Checkpoint Tracker for Kohya SD-Scripts
Tracks the best checkpoint during training based on loss metrics
Similar to CustomEvalSaveCallback in text training but adapted for image training
"""

import json
import os
import shutil
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BestCheckpointTracker:
    """
    Track best checkpoint during training and save it as the final output.
    
    This class monitors training loss per epoch and identifies the best checkpoint
    based on the lowest loss value. At the end of training, it copies the best
    checkpoint to the final output name (typically 'last.safetensors').
    
    Example:
        tracker = BestCheckpointTracker(output_dir="/path/to/output", output_name="last")
        
        # During training, after each epoch:
        for epoch in range(num_epochs):
            epoch_loss = loss_recorder.moving_average
            checkpoint_path = save_checkpoint(...)
            
            is_best = tracker.update(
                epoch=epoch + 1, 
                loss=epoch_loss, 
                checkpoint_path=checkpoint_path
            )
            
            if is_best:
                print(f"New best checkpoint at epoch {epoch+1}")
        
        # After training completes:
        tracker.save_best_as_final()
    """
    
    def __init__(
        self, 
        output_dir: str, 
        output_name: str = "last",
        save_metadata: bool = True,
        cleanup_old_checkpoints: bool = False,
        keep_n_checkpoints: int = 3
    ):
        """
        Initialize the best checkpoint tracker.
        
        Args:
            output_dir: Directory where checkpoints are saved
            output_name: Name for the final best checkpoint (without extension)
            save_metadata: Whether to save metadata JSON file
            cleanup_old_checkpoints: Whether to remove non-best checkpoints after training
            keep_n_checkpoints: Number of recent checkpoints to keep if cleanup is enabled
        """
        self.output_dir = output_dir
        self.output_name = output_name
        self.save_metadata = save_metadata
        self.cleanup_old_checkpoints = cleanup_old_checkpoints
        self.keep_n_checkpoints = keep_n_checkpoints
        
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_checkpoint_path = None
        self.loss_history: List[Dict] = []
        
        logger.info(f"BestCheckpointTracker initialized: output_dir={output_dir}, output_name={output_name}")
    
    def update(
        self, 
        epoch: int, 
        loss: float, 
        checkpoint_path: Optional[str] = None,
        global_step: Optional[int] = None
    ) -> bool:
        """
        Update tracker with new epoch information.
        
        Args:
            epoch: Current epoch number (1-indexed)
            loss: Loss value for this epoch (typically moving average)
            checkpoint_path: Path to the checkpoint file saved for this epoch
            global_step: Optional global training step
        
        Returns:
            bool: True if this is a new best checkpoint, False otherwise
        """
        # Record this epoch's information
        epoch_info = {
            'epoch': epoch,
            'loss': loss,
            'checkpoint': checkpoint_path
        }
        
        if global_step is not None:
            epoch_info['global_step'] = global_step
        
        self.loss_history.append(epoch_info)
        
        # Check if this is the best checkpoint so far
        is_best = False
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_checkpoint_path = checkpoint_path
            is_best = True
            
            logger.info(
                f"✨ New best checkpoint! Epoch {epoch} with loss {loss:.6f}"
                + (f" (step {global_step})" if global_step else "")
            )
        else:
            logger.info(
                f"Epoch {epoch} loss: {loss:.6f} "
                f"(best: {self.best_loss:.6f} at epoch {self.best_epoch})"
            )
        
        return is_best
    
    def save_best_as_final(self, force_sync_upload: bool = False) -> bool:
        """
        Copy the best checkpoint to the final output name.
        
        This should be called after training completes. It will:
        1. Copy the best checkpoint to {output_name}.safetensors
        2. Save metadata about the best checkpoint
        3. Optionally cleanup old checkpoints
        
        Args:
            force_sync_upload: Whether to force synchronous upload (for HuggingFace)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.best_checkpoint_path:
            logger.warning("No best checkpoint found to save as final")
            return False
        
        if not os.path.exists(self.best_checkpoint_path):
            logger.error(
                f"Best checkpoint file not found: {self.best_checkpoint_path}. "
                f"Cannot save as final checkpoint."
            )
            return False
        
        # Determine final checkpoint path
        # Support both .safetensors and .ckpt extensions
        checkpoint_ext = os.path.splitext(self.best_checkpoint_path)[1]
        final_checkpoint_name = f"{self.output_name}{checkpoint_ext}"
        final_checkpoint_path = os.path.join(self.output_dir, final_checkpoint_name)
        
        try:
            # Copy best checkpoint to final name
            logger.info(
                f"Copying best checkpoint from epoch {self.best_epoch} "
                f"(loss: {self.best_loss:.6f}) to {final_checkpoint_name}"
            )
            shutil.copy2(self.best_checkpoint_path, final_checkpoint_path)
            logger.info(f"✅ Best checkpoint saved as: {final_checkpoint_path}")
            
            # Save metadata
            if self.save_metadata:
                self._save_metadata()
            
            # Cleanup old checkpoints if requested
            if self.cleanup_old_checkpoints:
                self._cleanup_checkpoints()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving best checkpoint: {e}")
            return False
    
    def _save_metadata(self):
        """Save metadata about training and best checkpoint."""
        metadata_path = os.path.join(self.output_dir, "best_checkpoint_info.json")
        
        metadata = {
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'best_checkpoint': self.best_checkpoint_path,
            'total_epochs': len(self.loss_history),
            'all_losses': self.loss_history
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    def _cleanup_checkpoints(self):
        """
        Remove old checkpoints, keeping only:
        1. The best checkpoint
        2. The last N checkpoints (configurable)
        """
        if len(self.loss_history) <= self.keep_n_checkpoints:
            logger.info("Not enough checkpoints to cleanup")
            return
        
        # Determine which checkpoints to keep
        checkpoints_to_keep = set()
        
        # Keep the best checkpoint
        if self.best_checkpoint_path:
            checkpoints_to_keep.add(self.best_checkpoint_path)
        
        # Keep the last N checkpoints
        recent_checkpoints = [
            entry['checkpoint'] 
            for entry in self.loss_history[-self.keep_n_checkpoints:]
            if entry.get('checkpoint')
        ]
        checkpoints_to_keep.update(recent_checkpoints)
        
        # Remove checkpoints not in keep list
        removed_count = 0
        for entry in self.loss_history:
            checkpoint_path = entry.get('checkpoint')
            if checkpoint_path and checkpoint_path not in checkpoints_to_keep:
                if os.path.exists(checkpoint_path):
                    try:
                        os.remove(checkpoint_path)
                        removed_count += 1
                        logger.info(f"Removed old checkpoint: {checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {checkpoint_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old checkpoint(s)")
    
    def get_best_info(self) -> Dict:
        """
        Get information about the best checkpoint.
        
        Returns:
            dict: Dictionary with best_epoch, best_loss, best_checkpoint_path
        """
        return {
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'best_checkpoint_path': self.best_checkpoint_path,
            'total_epochs': len(self.loss_history)
        }
    
    def has_best_checkpoint(self) -> bool:
        """Check if a best checkpoint has been identified."""
        return self.best_epoch > 0 and self.best_checkpoint_path is not None

