from django import forms

class location(forms.Form):
	latlon=forms.CharField(label="",max_length=250,widget=forms.TextInput(attrs={'id': "search-field",'placeholder':"location"}))
	
	