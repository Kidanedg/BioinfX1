users = {
    "student": "1234",
    "admin": "admin"
}

def login(username, password):
    return users.get(username) == password
