from fastapi import APIRouter, Depends, status
from schemas.user import TokenResponse, UserLoginRequest, UserRegisterRequest, UserRegisterResponse
from services.user_service import UserService, get_user_service

router = APIRouter(prefix="/users", tags=["users"])


@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Creates a user account with a unique email and securely hashed password.",
    responses={
        201: {"description": "User created"},
        409: {"description": "Email already registered"},
        422: {"description": "Validation error"},
    },
)
async def register_user(
        payload: UserRegisterRequest,
        service: UserService = Depends(get_user_service),
) -> UserRegisterResponse:
    return await service.register(payload)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Login user",
    description="Authenticates user credentials and returns a JWT access token.",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"},
    },
)
async def login_user(
        payload: UserLoginRequest,
        service: UserService = Depends(get_user_service),
) -> TokenResponse:
    return await service.login(payload)
