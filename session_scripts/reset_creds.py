import sys
sys.path.insert(0, r"c:\Users\RoyVivasi\Documents\notebook")
from webapp import auth

NEW = {"admin": "zfxadmin", "test": "zfxtest"}
with auth.connect() as c:
    for username, pw in NEW.items():
        row = c.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            print(f"  {username}: NOT FOUND (skipped)")
            continue
        c.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                  (auth.hash_password(pw), username))
        print(f"  {username}: reset OK")
    c.commit()

# verify they authenticate
for username, pw in NEW.items():
    u = auth.authenticate(username, pw)
    print(f"  verify {username}/{pw}:", "OK" if u else "FAILED")
