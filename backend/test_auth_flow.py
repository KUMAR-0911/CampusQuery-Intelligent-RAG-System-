from unittest.mock import MagicMock
from fastapi import Response, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from main import login
from models import UserManager, UserStatus, UserRole
import auth

def test_login_function_directly():
    print("Testing login handler directly...")
    mock_um = MagicMock(spec=UserManager)
    raw_password = "SecretPassword123!"
    hashed_pwd = auth.get_password_hash(raw_password)
    
    mock_um.get_user_by_email.return_value = {
        "id": 42,
        "email": "test@student.edu",
        "hashed_password": hashed_pwd,
        "name": "Test Student",
        "nationality": "Indian",
        "role": UserRole.USER.value,
        "status": UserStatus.ACTIVE.value
    }
    
    # 1. Valid login test
    response = Response()
    form = OAuth2PasswordRequestForm(
        grant_type="password",
        username="test@student.edu",
        password=raw_password,
        scope="",
        client_id=None,
        client_secret=None
    )
    
    result = login(response=response, form_data=form, um=mock_um)
    assert "access_token" in result
    assert result["token_type"] == "bearer"
    assert result["user"]["email"] == "test@student.edu"
    assert result["user"]["name"] == "Test Student"
    assert "refresh_token=" in str(response.headers.get("set-cookie", ""))
    print("[PASS] Valid credentials returned tokens, user object, and refresh cookie.")

    # 2. Invalid password test
    form_wrong = OAuth2PasswordRequestForm(
        grant_type="password",
        username="test@student.edu",
        password="WrongPassword!",
        scope="",
        client_id=None,
        client_secret=None
    )
    try:
        login(response=Response(), form_data=form_wrong, um=mock_um)
        assert False, "Should have raised HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        assert e.detail == "Incorrect email or password"
        print("[PASS] Invalid password raised 400 with 'Incorrect email or password'.")

    # 3. Nonexistent user test
    mock_um.get_user_by_email.return_value = None
    try:
        login(response=Response(), form_data=form, um=mock_um)
        assert False, "Should have raised HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        print("[PASS] Nonexistent user raised 400.")

    # 4. Inactive user test
    mock_um.get_user_by_email.return_value = {
        "id": 43,
        "email": "pending@student.edu",
        "hashed_password": hashed_pwd,
        "name": "Pending Student",
        "nationality": "Indian",
        "role": UserRole.USER.value,
        "status": UserStatus.PENDING_VERIFICATION.value
    }
    try:
        login(response=Response(), form_data=form, um=mock_um)
        assert False, "Should have raised HTTPException 403"
    except HTTPException as e:
        assert e.status_code == 403
        assert "PENDING_VERIFICATION" in e.detail
        print("[PASS] Pending verification user raised 403.")

    print("\nALL UNIT AND FUNCTIONAL TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_login_function_directly()
