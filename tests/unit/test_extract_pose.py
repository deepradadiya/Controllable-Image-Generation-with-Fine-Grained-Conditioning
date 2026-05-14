"""
Unit tests for pose skeleton extraction module

Tests the pose extraction functionality including DWPose integration,
MediaPipe fallback, and output validation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.extract_pose import (
    PoseKeypoint, PoseDetection, PoseFormat,
    MediaPipePoseExtractor, PoseExtractor,
    create_pose_extractor, extract_pose_from_image
)


class TestPoseKeypoint(unittest.TestCase):
    """Test PoseKeypoint class"""
    
    def test_keypoint_creation(self):
        """Test keypoint creation and basic properties"""
        kp = PoseKeypoint(x=0.5, y=0.3, confidence=0.8)
        
        self.assertEqual(kp.x, 0.5)
        self.assertEqual(kp.y, 0.3)
        self.assertEqual(kp.confidence, 0.8)
        self.assertTrue(kp.visible)
    
    def test_keypoint_validation(self):
        """Test keypoint validation logic"""
        # Valid keypoint
        valid_kp = PoseKeypoint(x=0.5, y=0.3, confidence=0.8)
        self.assertTrue(valid_kp.is_valid())
        
        # Invalid coordinates
        invalid_x = PoseKeypoint(x=1.5, y=0.3, confidence=0.8)
        self.assertFalse(invalid_x.is_valid())
        
        invalid_y = PoseKeypoint(x=0.5, y=-0.1, confidence=0.8)
        self.assertFalse(invalid_y.is_valid())
        
        # Low confidence
        low_conf = PoseKeypoint(x=0.5, y=0.3, confidence=0.05)
        self.assertFalse(low_conf.is_valid())
        
        # Not visible
        not_visible = PoseKeypoint(x=0.5, y=0.3, confidence=0.8, visible=False)
        self.assertFalse(not_visible.is_valid())
    
    def test_to_tuple(self):
        """Test conversion to tuple"""
        kp = PoseKeypoint(x=0.5, y=0.3, confidence=0.8)
        self.assertEqual(kp.to_tuple(), (0.5, 0.3, 0.8))


class TestPoseDetection(unittest.TestCase):
    """Test PoseDetection class"""
    
    def setUp(self):
        """Set up test pose detection"""
        self.valid_keypoints = [
            PoseKeypoint(x=0.5, y=0.3, confidence=0.8),
            PoseKeypoint(x=0.6, y=0.4, confidence=0.7),
            PoseKeypoint(x=0.4, y=0.5, confidence=0.9),
            PoseKeypoint(x=0.3, y=0.6, confidence=0.6),
            PoseKeypoint(x=0.7, y=0.2, confidence=0.8)
        ]
        
        self.invalid_keypoints = [
            PoseKeypoint(x=1.5, y=0.3, confidence=0.8),  # Invalid x
            PoseKeypoint(x=0.6, y=0.4, confidence=0.05)  # Low confidence
        ]
    
    def test_pose_detection_creation(self):
        """Test pose detection creation"""
        pose = PoseDetection(
            keypoints=self.valid_keypoints,
            pose_confidence=0.8,
            format=PoseFormat.COCO_17
        )
        
        self.assertEqual(len(pose.keypoints), 5)
        self.assertEqual(pose.pose_confidence, 0.8)
        self.assertEqual(pose.format, PoseFormat.COCO_17)
    
    def test_get_valid_keypoints(self):
        """Test filtering of valid keypoints"""
        all_keypoints = self.valid_keypoints + self.invalid_keypoints
        pose = PoseDetection(keypoints=all_keypoints)
        
        valid_kps = pose.get_valid_keypoints()
        self.assertEqual(len(valid_kps), 5)  # Only valid keypoints
    
    def test_has_sufficient_keypoints(self):
        """Test sufficient keypoints check"""
        # Sufficient keypoints
        pose_sufficient = PoseDetection(keypoints=self.valid_keypoints)
        self.assertTrue(pose_sufficient.has_sufficient_keypoints(min_keypoints=3))
        
        # Insufficient keypoints
        pose_insufficient = PoseDetection(keypoints=self.valid_keypoints[:2])
        self.assertFalse(pose_insufficient.has_sufficient_keypoints(min_keypoints=5))


class TestMediaPipePoseExtractor(unittest.TestCase):
    """Test MediaPipe pose extractor"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image = Image.new('RGB', (512, 512), color='white')
        self.test_array = np.ones((512, 512, 3), dtype=np.uint8) * 255
    
    @patch('mediapipe.solutions.pose.Pose')
    def test_mediapipe_initialization(self, mock_pose_class):
        """Test MediaPipe extractor initialization"""
        mock_pose_instance = Mock()
        mock_pose_class.return_value = mock_pose_instance
        
        extractor = MediaPipePoseExtractor()
        
        self.assertIsNotNone(extractor.pose)
        mock_pose_class.assert_called_once()
    
    @patch('mediapipe.solutions.pose.Pose')
    def test_extract_pose_no_detection(self, mock_pose_class):
        """Test pose extraction when no pose is detected"""
        mock_pose_instance = Mock()
        mock_pose_instance.process.return_value.pose_landmarks = None
        mock_pose_class.return_value = mock_pose_instance
        
        extractor = MediaPipePoseExtractor()
        poses = extractor.extract_pose(self.test_image)
        
        self.assertEqual(len(poses), 0)
    
    @patch('mediapipe.solutions.pose.Pose')
    def test_extract_pose_with_detection(self, mock_pose_class):
        """Test pose extraction with successful detection"""
        # Mock MediaPipe results
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.3
        mock_landmark.visibility = 0.8
        
        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33  # MediaPipe has 33 landmarks
        
        mock_pose_instance = Mock()
        mock_pose_instance.process.return_value = mock_results
        mock_pose_class.return_value = mock_pose_instance
        
        extractor = MediaPipePoseExtractor()
        poses = extractor.extract_pose(self.test_image)
        
        self.assertEqual(len(poses), 1)
        self.assertEqual(len(poses[0].keypoints), 33)
        self.assertEqual(poses[0].format, PoseFormat.MEDIAPIPE_33)
    
    def test_validate_output(self):
        """Test output validation"""
        extractor = MediaPipePoseExtractor()
        
        # Valid output
        valid_output = np.ones((512, 512, 3), dtype=np.uint8)
        self.assertTrue(extractor.validate_output(valid_output))
        
        # Invalid outputs
        self.assertFalse(extractor.validate_output(None))
        self.assertFalse(extractor.validate_output(np.ones((512, 512))))  # Wrong dimensions
        self.assertFalse(extractor.validate_output(np.ones((10, 10, 3))))  # Too small


class TestPoseExtractor(unittest.TestCase):
    """Test main PoseExtractor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_image = Image.new('RGB', (512, 512), color='white')
    
    @patch('data.extract_pose.MediaPipePoseExtractor')
    @patch('data.extract_pose.DWPoseExtractor')
    def test_initialization_with_dwpose(self, mock_dwpose, mock_mediapipe):
        """Test initialization with DWPose preference"""
        mock_dwpose.return_value = Mock()
        mock_mediapipe.return_value = Mock()
        
        extractor = PoseExtractor(prefer_dwpose=True, speed_critical=False)
        
        mock_dwpose.assert_called_once()
        mock_mediapipe.assert_called_once()
    
    @patch('data.extract_pose.MediaPipePoseExtractor')
    def test_initialization_speed_critical(self, mock_mediapipe):
        """Test initialization in speed-critical mode"""
        mock_mediapipe.return_value = Mock()
        
        extractor = PoseExtractor(speed_critical=True)
        
        mock_mediapipe.assert_called_once()
        self.assertIsNone(extractor.dwpose_extractor)
    
    @patch('data.extract_pose.MediaPipePoseExtractor')
    def test_extract_with_mediapipe_fallback(self, mock_mediapipe):
        """Test extraction with MediaPipe fallback"""
        # Mock MediaPipe extractor
        mock_mp_instance = Mock()
        mock_mp_instance.extract_pose.return_value = [Mock()]
        mock_mp_instance.render_skeleton.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_mediapipe.return_value = mock_mp_instance
        
        extractor = PoseExtractor(prefer_dwpose=False)
        result = extractor.extract(self.test_image)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (512, 512, 3))
        mock_mp_instance.extract_pose.assert_called_once()
    
    @patch('data.extract_pose.MediaPipePoseExtractor')
    def test_extract_no_poses_detected(self, mock_mediapipe):
        """Test extraction when no poses are detected"""
        mock_mp_instance = Mock()
        mock_mp_instance.extract_pose.return_value = []  # No poses detected
        mock_mediapipe.return_value = mock_mp_instance
        
        extractor = PoseExtractor(prefer_dwpose=False)
        result = extractor.extract(self.test_image)
        
        # Should return blank image
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (512, 512, 3))
        # Should be mostly zeros (blank)
        self.assertTrue(np.all(result == 0))
    
    @patch('data.extract_pose.MediaPipePoseExtractor')
    def test_batch_extract(self, mock_mediapipe):
        """Test batch extraction"""
        mock_mp_instance = Mock()
        mock_mp_instance.extract_pose.return_value = [Mock()]
        mock_mp_instance.render_skeleton.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_mediapipe.return_value = mock_mp_instance
        
        extractor = PoseExtractor(prefer_dwpose=False)
        images = [self.test_image, self.test_image]
        results = extractor.batch_extract(images, show_progress=False)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (512, 512, 3))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_create_pose_extractor(self):
        """Test pose extractor creation"""
        # Normal mode
        extractor = create_pose_extractor(speed_critical=False)
        self.assertIsInstance(extractor, PoseExtractor)
        self.assertTrue(extractor.prefer_dwpose)
        
        # Speed critical mode
        extractor_fast = create_pose_extractor(speed_critical=True)
        self.assertIsInstance(extractor_fast, PoseExtractor)
        self.assertTrue(extractor_fast.speed_critical)
    
    @patch('data.extract_pose.PoseExtractor')
    def test_extract_pose_from_image(self, mock_extractor_class):
        """Test convenience function for single image extraction"""
        mock_extractor = Mock()
        mock_extractor.extract.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_extractor_class.return_value = mock_extractor
        
        test_image = Image.new('RGB', (512, 512), color='white')
        result = extract_pose_from_image(test_image)
        
        self.assertIsInstance(result, np.ndarray)
        mock_extractor.extract.assert_called_once()


if __name__ == '__main__':
    unittest.main()