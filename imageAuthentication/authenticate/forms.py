# authenticate/forms.py
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate
from django.core.validators import EmailValidator

class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={'placeholder': 'First Name'}))
    last_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={'placeholder': 'Last Name'}))
    email = forms.EmailField(required=True, validators=[EmailValidator()], widget=forms.EmailInput(attrs={'placeholder': 'Email Address'}))

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        email = self.cleaned_data['email']
        # Auto-generate username from email (before @)
        username = email.split('@')[0]
        # Make sure username is unique
        base_username = username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1
        
        user.username = username
        user.email = email
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
        return user

# REPLACE YOUR EXISTING FORM WITH THIS:
class EmailOrUsernameAuthenticationForm(AuthenticationForm):
    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username and password:
            # Try to authenticate with username first
            self.user_cache = authenticate(self.request, username=username, password=password)
            
            # If not found, try with email
            if self.user_cache is None:
                try:
                    # Get user by email
                    user_by_email = User.objects.get(email=username)
                    # Authenticate using the username from found email
                    self.user_cache = authenticate(self.request, username=user_by_email.username, password=password)
                except User.DoesNotExist:
                    pass
            
            # Handle invalid login
            if self.user_cache is None:
                raise forms.ValidationError(
                    "Invalid email/username or password.",
                    code='invalid_login',
                )
            else:
                self.confirm_login_allowed(self.user_cache)
        
        return self.cleaned_data
    
