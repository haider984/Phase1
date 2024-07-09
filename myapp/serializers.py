# serializers.py
from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'is_free', 'unique_id']
        extra_kwargs = {
            'password': {'write_only': True},
            'email': {'required': True},
            'is_free': {'required': False},
            'unique_id': {'read_only': True}
        }

    def validate_email(self, value):
        """
        Check if the email is already registered.
        """
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email address is already registered.")
        return value

    def create(self, validated_data):
        # Remove is_free from validated_data if it's present
        is_free = validated_data.pop('is_free', None)  # Default to True if not provided
        
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            is_free=is_free
        )
        user.is_active = False  # Deactivate account until email confirmation
        user.save()
        return user

class SignupSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email', 'password']

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()

class PasswordResetConfirmSerializer(serializers.Serializer):
    password = serializers.CharField(write_only=True)

class DocumentQuerySerializer(serializers.Serializer):
    file = serializers.FileField()
    user_id = serializers.CharField()
    document_name=serializers.CharField()

class ChatQuerySerializer(serializers.Serializer):
    query = serializers.CharField()
    document_name=serializers.CharField()

class PDFUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class ConversationHistorySerializer(serializers.Serializer):
    conversation_history = serializers.ListField(
        child=serializers.CharField()
    )

class ClearHistoryResponseSerializer(serializers.Serializer):
    status = serializers.CharField()


class CustomReportSerializer(serializers.Serializer):
    query = serializers.CharField()
    document_name=serializers.CharField()