from django.urls import path
from .views import (
    test_email,
    DocumentQueryView,
    index,
    ConversationHistoryView,
    PasswordResetConfirmView,
    ClearConversationHistoryView,
    ChatQueryView,
    PDFSummaryView,
    SignupView,
    ConfirmEmailView,
    LoginView,
    PasswordResetRequestView,
    TokenObtainPairView,
    TokenRefreshView,
    LogoutView,
    CustomReportView,
)
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)


urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('query-document/', DocumentQueryView.as_view(), name='query-document'),#
    path('', index, name='index'),  # Route root URL to index view
    path('summarize-pdf/', PDFSummaryView.as_view(), name='summarize-pdf'),#
    path('history/<str:user_id>/', ConversationHistoryView.as_view(), name='conversation_history'),
    path('clear-conversation-history/', ClearConversationHistoryView.as_view(), name='clear-conversation-history'),
    path('chat-query/<str:user_id>/', ChatQueryView.as_view(), name='chat-query'),#
    path('signup/', SignupView.as_view(), name='signup'),
    path('confirm-email/<str:uidb64>/<str:token>/', ConfirmEmailView.as_view(), name='confirm-email'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('reset-password/', PasswordResetRequestView.as_view(), name='reset-password-request'),
    path('reset-password-confirm/<str:uidb64>/<str:token>/', PasswordResetConfirmView.as_view(), name='reset-password-confirm'),
    path('test-email/', test_email, name='test-email'),

    path('CustomReport/<str:user_id>/', CustomReportView.as_view(), name='chat-query'),#

]
