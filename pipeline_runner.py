import os
from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from tactical_view_converter import TacticalViewConverter
from speed_and_distance_calculator import SpeedAndDistanceCalculator
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    CourtKeypointDrawer,
    TeamBallControlDrawer,
    FrameNumberDrawer,
    PassInterceptionDrawer,
    TacticalViewDrawer,
    SpeedAndDistanceDrawer,
)
from configs import (
    STUBS_DEFAULT_PATH,
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
)


def run_analysis(
    input_video_path: str,
    output_video_path: str = OUTPUT_VIDEO_PATH,
    stub_path: str = STUBS_DEFAULT_PATH,
    use_stubs: bool = True,
    court_image_path: str = "./images/basketball_court.png",
):
    """
    Execute the basketball analysis pipeline for a single input video.

    Args:
        input_video_path: Path to the source video file.
        output_video_path: Path where the annotated video will be written.
        stub_path: Directory to read/write cached detection and tracking results.
        use_stubs: Whether to read from/write to stub files to accelerate runs.
        court_image_path: Background image for the tactical view overlay.
    """
    video_frames = read_video(input_video_path)

    # Initialize components
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)
    court_keypoint_detector = CourtKeypointDetector(COURT_KEYPOINT_DETECTOR_PATH)

    # Run detectors with optional stubbing
    player_tracks = player_tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=os.path.join(stub_path, "player_track_stubs.pkl"),
    )

    ball_tracks = ball_tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=os.path.join(stub_path, "ball_track_stubs.pkl"),
    )

    court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=os.path.join(stub_path, "court_key_points_stub.pkl"),
    )

    # Refine ball tracks
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(
        video_frames,
        player_tracks,
        read_from_stub=use_stubs,
        stub_path=os.path.join(stub_path, "player_assignment_stub.pkl"),
    )

    # Possession and events
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition = ball_aquisition_detector.detect_ball_possession(
        player_tracks, ball_tracks
    )

    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(
        ball_aquisition, player_assignment
    )
    interceptions = pass_and_interception_detector.detect_interceptions(
        ball_aquisition, player_assignment
    )

    # Tactical view conversion
    tactical_view_converter = TacticalViewConverter(court_image_path=court_image_path)
    court_keypoints_per_frame = tactical_view_converter.validate_keypoints(
        court_keypoints_per_frame
    )
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(
        court_keypoints_per_frame, player_tracks
    )

    # Speed and distance
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.actual_width_in_meters,
        tactical_view_converter.actual_height_in_meters,
    )
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(
        tactical_player_positions
    )
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(
        player_distances_per_frame
    )

    # Draw overlays
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    frame_number_drawer = FrameNumberDrawer()
    pass_and_interceptions_drawer = PassInterceptionDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    speed_and_distance_drawer = SpeedAndDistanceDrawer()

    output_video_frames = player_tracks_drawer.draw(
        video_frames, player_tracks, player_assignment, ball_aquisition
    )
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = court_keypoint_drawer.draw(
        output_video_frames, court_keypoints_per_frame
    )
    output_video_frames = frame_number_drawer.draw(output_video_frames)
    output_video_frames = team_ball_control_drawer.draw(
        output_video_frames, player_assignment, ball_aquisition
    )
    output_video_frames = pass_and_interceptions_drawer.draw(
        output_video_frames, passes, interceptions
    )
    output_video_frames = speed_and_distance_drawer.draw(
        output_video_frames,
        player_tracks,
        player_distances_per_frame,
        player_speed_per_frame,
    )
    output_video_frames = tactical_view_drawer.draw(
        output_video_frames,
        tactical_view_converter.court_image_path,
        tactical_view_converter.width,
        tactical_view_converter.height,
        tactical_view_converter.key_points,
        tactical_player_positions,
        player_assignment,
        ball_aquisition,
    )

    save_video(output_video_frames, output_video_path)

