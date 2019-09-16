DIFERENCIA ENTRE UN INT Y UN UNINT

Un entero con signo de 32 bits es un entero cuyo valor se representa en 32 bits (es decir, 4 bytes). Los bits son binarios, lo que significa que solo pueden ser cero o uno. Entonces, el entero con signo de 32 bits es una cadena de 32 ceros y unos.

La parte firmada del entero se refiere a su capacidad de representar valores positivos y negativos. Un entero positivo tendr� su bit m�s significativo (el bit inicial) ser� un cero, mientras que un entero negativo tendr� su bit m�s significativo como uno. Debido a esto, el bit m�s significativo de un entero con signo se denomina t�picamente "bit de signo" ya que su prop�sito es denotar el signo del entero.

CUAL ES EL TIPO DEFAULT BASE PYTHON? NUMPY? PYTORCH?
PARA INT

PYTHON
   Python 3: 32 bits


NUMPY
   32 bits integer

PYTORCH
   32-bit integer (signed)

CUAL ES EL RANGO DE VALORES QUE CADA UNO DE ESTOS TIPOS PUEDE REPRESENTAR?

torch.float32 o torch.float

-3.4E+38 a +3.4E+38

torch.float64 o torch.double

-1.7E+308 a +1.7E+308

torch.float16 o torch.half

32768 a 65536

torch.int8

-128 a 127

torch.uint8

0 to 255

torch.int16 o torch.short


