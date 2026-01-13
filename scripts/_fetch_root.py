import urllib.request
u = urllib.request.urlopen('http://127.0.0.1:8000')
print('status', u.getcode())
body = u.read().decode('utf-8')
print(body[:400])
