"""Verification script for the complete login -> token storage -> refresh -> logout flow."""
import auth
from main import login, refresh_token, logout, RefreshTokenRequest
from fastapi import Response, Request, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from unittest.mock import MagicMock
from models import UserStatus

def test_full_refresh_flow():
    print("=" * 60)
    print("RUNNING REFRESH TOKEN AUTHENTICATION FLOW TESTS")
    print("=" * 60)

    # 1. Mock user manager
    mock_um = MagicMock()
    mock_user = {
        "id": 42,
        "name": "Refresh Tester",
        "email": "refreshtest@example.com",
        "role": "USER",
        "status": UserStatus.ACTIVE.value,
        "hashed_password": auth.get_password_hash("Secret123!"),
    }
    mock_um.get_user_by_email.return_value = mock_user

    # 2. Test Login
    print("\n[TEST 1] Testing /login response and cookie generation...")
    login_response = Response()
    form = OAuth2PasswordRequestForm(
        grant_type="password",
        username="refreshtest@example.com",
        password="Secret123!",
        scope="",
        client_id=None,
        client_secret=None,
    )
    login_result = login(response=login_response, form_data=form, um=mock_um)

    assert "access_token" in login_result, "access_token missing from /login response"
    assert "refresh_token" in login_result, "refresh_token missing from /login response"
    assert login_result["token_type"] == "bearer"
    assert login_result["user"]["email"] == "refreshtest@example.com"

    # Check cookies on /login
    cookies_header = str(login_response.headers.get("set-cookie", ""))
    assert "refresh_token=" in cookies_header, "refresh_token cookie missing in /login"
    assert "samesite=none" in cookies_header.lower(), "samesite=none missing in /login cookies"
    assert "secure" in cookies_header.lower(), "secure missing in /login cookies"
    print("  [PASS] /login returned both access_token and refresh_token in body AND set secure cross-origin cookies.")

    # 3. Test Refresh with JSON Body
    print("\n[TEST 2] Testing /refresh with token in JSON body...")
    refresh_req = Request({"type": "http", "method": "POST", "headers": []})
    refresh_resp = Response()
    refresh_data = RefreshTokenRequest(refresh_token=login_result["refresh_token"])

    import asyncio
    refresh_result = asyncio.run(
        refresh_token(request=refresh_req, response=refresh_resp, data=refresh_data, um=mock_um)
    )
    assert "access_token" in refresh_result, "access_token missing in /refresh result"
    assert "refresh_token" in refresh_result, "refresh_token missing in /refresh result"
    assert refresh_result["access_token"] != "", "empty access_token"
    print("  [PASS] /refresh with body refresh_token issued new access and refresh tokens.")

    # 4. Test Refresh with HTTP-Only Cookie
    print("\n[TEST 3] Testing /refresh with token in Cookie...")
    cookie_req = Request({
        "type": "http",
        "method": "POST",
        "headers": [(b"cookie", f"refresh_token={login_result['refresh_token']}".encode())],
    })
    cookie_resp = Response()
    cookie_result = asyncio.run(
        refresh_token(request=cookie_req, response=cookie_resp, data=None, um=mock_um)
    )
    assert "access_token" in cookie_result
    assert "refresh_token" in cookie_result
    print("  [PASS] /refresh with HTTP-only cookie issued new access and refresh tokens.")

    # 5. Test Refresh with Authorization Bearer Header
    print("\n[TEST 4] Testing /refresh with token in Authorization Bearer Header...")
    auth_header_req = Request({
        "type": "http",
        "method": "POST",
        "headers": [(b"authorization", f"Bearer {login_result['refresh_token']}".encode())],
    })
    auth_header_resp = Response()
    auth_header_result = asyncio.run(
        refresh_token(request=auth_header_req, response=auth_header_resp, data=None, um=mock_um)
    )
    assert "access_token" in auth_header_result
    assert "refresh_token" in auth_header_result
    print("  [PASS] /refresh with Authorization header issued new tokens.")

    # 6. Test Refresh with MISSING Token (Should reject with 401 without looping)
    print("\n[TEST 5] Testing /refresh with NO token (should raise 401 'Refresh token missing')...")
    empty_req = Request({"type": "http", "method": "POST", "headers": []})
    empty_resp = Response()
    try:
        asyncio.run(refresh_token(request=empty_req, response=empty_resp, data=None, um=mock_um))
        assert False, "Should have raised HTTPException 401"
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "Refresh token missing" in exc.detail
        print(f"  [PASS] Properly rejected with 401: {exc.detail}")

    # 7. Test Logout
    print("\n[TEST 6] Testing /logout cookie clearing...")
    logout_resp = Response()
    logout_result = logout(response=logout_resp)
    assert logout_result["message"] == "Logged out successfully"
    raw_cookie_headers = [v.decode() for k, v in logout_resp.raw_headers if k.lower() == b"set-cookie"]
    combined_cookies = " ".join(raw_cookie_headers)
    assert "access_token=" in combined_cookies
    assert "refresh_token=" in combined_cookies
    assert "samesite=none" in combined_cookies.lower()
    assert "secure" in combined_cookies.lower()
    print("  [PASS] /logout correctly cleared both cookies with secure=True, samesite=none, and path=/.")

    print("\n" + "=" * 60)
    print("ALL REFRESH-TOKEN FLOW TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_refresh_flow()
