DIFERENCIA ENTRE UN INT Y UN UNINT

Un entero con signo de 32 bits es un entero cuyo valor se representa en 32 bits (es decir, 4 bytes). Los bits son binarios, lo que significa que solo pueden ser cero o uno. Entonces, el entero con signo de 32 bits es una cadena de 32 ceros y unos.

La parte firmada del entero se refiere a su capacidad de representar valores positivos y negativos. Un entero positivo tendrá su bit más significativo (el bit inicial) será un cero, mientras que un entero negativo tendrá su bit más significativo como uno. Debido a esto, el bit más significativo de un entero con signo se denomina típicamente "bit de signo" ya que su propósito es denotar el signo del entero.

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

-32768 
32767

torch.int32 o torch.int

2,147,483,647
-2,147,483,648

torch.int64 o torch.long

-9,223,372,036,854,775,808 
9,223,372,036,854,775,807

NATHALIA MORALES

