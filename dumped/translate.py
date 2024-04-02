# This would contain the code to translate the text from our language to hindi and english

from googletrans import Translator
trans= Translator()
str = 'स्थापना सफलतापूर्वक पूरी हुई है।'
output = trans.translate(str, dest='en')
print(output.text)