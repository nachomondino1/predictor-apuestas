from django.shortcuts import render
from django.http import HttpResponse

# A view function is a function that takes a request and returns a response. 
# It's a request handler and in some framewroks is called an action.

def say_hello(request):
    # return HttpResponse('Hello World')
    x = 1
    x = 3
    return render(request, 'hello.html', {'name': 'Carolina'})
