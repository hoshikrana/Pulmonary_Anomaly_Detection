/**
 * auth.js — Authentication handling (login/signup)
 */

const loginForm = document.getElementById('login-form');
if (loginForm) {
  loginForm.addEventListener('submit', handleLogin);
}

const signupForm = document.getElementById('signup-form');
if (signupForm) {
  signupForm.addEventListener('submit', handleSignup);
}

async function handleLogin(e) {
  e.preventDefault();

  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;

  try {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      alert('Login failed: ' + (data.error || 'Unknown error'));
      return;
    }

    // Store token (in production, use secure HTTPOnly cookie)
    localStorage.setItem('auth_token', data.token);
    localStorage.setItem('user_email', email);

    alert(data.message);
    window.location.href = '/dashboard';
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

async function handleSignup(e) {
  e.preventDefault();

  const name = document.getElementById('name').value;
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  const confirmPassword = document.getElementById('confirm-password').value;

  // Validate password
  if (password !== confirmPassword) {
    alert('Passwords do not match');
    return;
  }

  if (password.length < 8) {
    alert('Password must be at least 8 characters');
    return;
  }

  if (!/[A-Z]/.test(password) || !/[0-9]/.test(password)) {
    alert('Password must contain uppercase letter and number');
    return;
  }

  try {
    const response = await fetch('/api/auth/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name, email, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      alert('Signup failed: ' + (data.error || 'Unknown error'));
      return;
    }

    alert(data.message);
    window.location.href = '/auth/login';
  } catch (error) {
    alert('Error: ' + error.message);
  }
}

// Check authentication status
function checkAuth() {
  const token = localStorage.getItem('auth_token');
  if (!token && window.location.pathname !== '/auth/login' && window.location.pathname !== '/auth/signup') {
    // Redirect to login if not authenticated and not on auth page
    // For demo, we'll allow access
  }
}

document.addEventListener('DOMContentLoaded', checkAuth);
