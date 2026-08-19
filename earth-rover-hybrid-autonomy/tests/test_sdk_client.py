import base64

import cv2
import numpy as np
import pytest
import requests

from earth_rover.core.types import ControlCommand
from earth_rover.sdk_client import EarthRoverSDKClient, SDKClientError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.content = b"{}"

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        path = "/" + url.split("/", 3)[3]
        self.calls.append((method, path, kwargs))
        return FakeResponse(self.routes[(method, path)])


def encoded_image():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buffer).decode("ascii")


def client_with(routes):
    client = EarthRoverSDKClient("http://localhost:8000", 0.25)
    client.session = FakeSession(routes)
    return client


def test_front_frame_uses_v2_front_and_front_frame_key():
    client = client_with({("GET", "/v2/front"): {"front_frame": encoded_image(), "timestamp": 1.0}})

    frame = client.get_front_frame()

    assert frame.image.shape == (3, 4, 3)
    assert frame.sdk_timestamp == 1.0
    assert client.session.calls[0][0:2] == ("GET", "/v2/front")


def test_front_frame_returns_last_good_frame_after_transient_sdk_failure():
    first = client_with(
        {("GET", "/v2/front"): {"front_frame": encoded_image(), "timestamp": 1.0}}
    )
    cached = first.get_front_frame()
    first.session = FakeSession({})

    recovered = first.get_front_frame()

    assert recovered is not cached
    assert recovered.timestamp == cached.timestamp
    assert recovered.source_frame_new is False
    assert recovered.sdk_timestamp == 1.0
    assert [call[0:2] for call in first.session.calls] == [
        ("GET", "/v2/front"),
        ("GET", "/front"),
    ]


def test_duplicate_source_frame_id_preserves_first_seen_age(monkeypatch):
    payload = {
        "front_frame": encoded_image(),
        "timestamp": 100.0,
        "source_frame_id": "session:1000:42:10.0",
        "source_media_time_sec": 10.0,
    }
    client = client_with({("GET", "/v2/front"): payload})
    timestamps = iter((10.0, 20.0, 30.0))
    monkeypatch.setattr("earth_rover.sdk_client.time.time", lambda: next(timestamps))

    first = client.get_front_frame()
    duplicate = client.get_front_frame()
    client.session.routes[("GET", "/v2/front")] = {
        **payload,
        "source_frame_id": "session:1000:43:10.1",
        "source_media_time_sec": 10.1,
    }
    fresh = client.get_front_frame()

    assert first.source_frame_new is True
    assert duplicate.source_frame_new is False
    assert duplicate.timestamp == 10.0
    assert duplicate.source_media_time_sec == 10.0
    assert fresh.source_frame_new is True
    assert fresh.timestamp == 30.0


def test_mission_and_checkpoint_endpoints_match_official_sdk():
    routes = {
        ("POST", "/start-mission"): {"message": "Mission started successfully"},
        ("GET", "/checkpoints-list"): {
            "checkpoints_list": [{"sequence": 1, "latitude": "30.48243713", "longitude": "114.3026428"}],
            "latest_scanned_checkpoint": 0,
        },
        ("POST", "/checkpoint-reached"): {"message": "Checkpoint reached successfully"},
        ("POST", "/end-mission"): {"message": "Mission ended successfully"},
        ("GET", "/mission-status"): {"mission_active": True},
    }
    client = client_with(routes)

    assert client.start_mission() is True
    assert client.get_checkpoints()[0]["sequence"] == 1
    assert client.get_checkpoint_state()["latest_scanned_checkpoint"] == 0.0
    assert client.report_checkpoint() is True
    assert client.end_mission() is True
    assert client.get_mission_status()["mission_active"] is True

    assert [call[0:2] for call in client.session.calls] == [
        ("POST", "/start-mission"),
        ("GET", "/checkpoints-list"),
        ("GET", "/checkpoints-list"),
        ("POST", "/checkpoint-reached"),
        ("POST", "/end-mission"),
        ("GET", "/mission-status"),
    ]


class ErrorResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body
        self.content = body.encode()

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def json(self):
        return {}


class FailingSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, timeout, kwargs))
        return self.response


def test_checkpoint_report_failure_surfaces_response_body():
    client = EarthRoverSDKClient("http://localhost:8000", 1.0)
    session = FailingSession(ErrorResponse(422, '{"error":"expected checkpoint 3"}'))
    client.session = session

    with pytest.raises(SDKClientError) as excinfo:
        client.report_checkpoint_details()

    assert "expected checkpoint 3" in str(excinfo.value)


def test_checkpoint_report_uses_longer_timeout_than_control_loop_calls():
    client = EarthRoverSDKClient("http://localhost:8000", 1.0)
    session = FailingSession(ErrorResponse(422, "{}"))
    client.session = session

    with pytest.raises(SDKClientError):
        client.report_checkpoint_details()

    assert client.checkpoint_timeout >= 5.0
    assert session.calls[0][2] == client.checkpoint_timeout
    assert session.calls[0][2] > client.timeout


def test_cached_mission_route_is_read_only() -> None:
    client = client_with(
        {
            ("GET", "/mission-route"): {
                "checkpoints_list": [{"sequence": 1, "latitude": 30.1, "longitude": 114.1}],
                "latest_scanned_checkpoint": 0,
                "mission_active": True,
                "route_loaded": True,
            }
        }
    )

    route = client.get_mission_route()

    assert route["route_loaded"] is True
    assert route["checkpoints"][0]["sequence"] == 1
    assert [call[0:2] for call in client.session.calls] == [("GET", "/mission-route")]


def test_control_payload_matches_official_sdk():
    client = client_with({("POST", "/control"): {"message": "Command sent successfully"}})

    assert client.send_control(ControlCommand(0.1, -0.2, lamp=1)) is True

    method, path, kwargs = client.session.calls[0]
    assert (method, path) == ("POST", "/control")
    assert kwargs["json"] == {"command": {"linear": 0.1, "angular": -0.2, "lamp": 1}}


def test_data_parses_official_nested_rpm_shape():
    client = client_with(
        {
            ("GET", "/data"): {
                "battery": 100,
                "signal_level": 5,
                "orientation": 128,
                "speed": 0,
                "gps_signal": 31.25,
                "latitude": 22.753774642944336,
                "longitude": 114.09095001220703,
                "timestamp": 1724189733.208559,
                "rpms": [
                    [1, 2, 3, 4, 1725434567.194],
                    [5, 6, 7, 8, 1725434597.726],
                ],
            }
        }
    )

    data = client.get_data()

    assert data.latitude == 22.753774642944336
    assert data.longitude == 114.09095001220703
    assert data.orientation == 128
    assert data.rpms == [5.0, 6.0, 7.0, 8.0]
    assert data.sdk_timestamp == 1724189733.208559
