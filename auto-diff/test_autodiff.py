import unittest
import dual_autodiff 

class TestAutoDiff(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		print('... <<< Testing >>> ...')

	def test_add(self): 
		x = dual_autodiff.Dual(1, 2)
		y = dual_autodiff.Dual(2, 3)
		z = dual_autodiff.Dual(1+2, 2+3)
		self.assertEqual((x+y).real, z.real)
		self.assertEqual((x+y).dual, z.dual)

	def test_sub(self): 
		x = dual_autodiff.Dual(1, 2)
		y = dual_autodiff.Dual(2, 3)
		z = dual_autodiff.Dual(1-2, 2-3)
		self.assertEqual((x-y).real, z.real)
		self.assertEqual((x-y).dual, z.dual)

	def test_mul(self):
		x = dual_autodiff.Dual(1, -2)
		y = dual_autodiff.Dual(-2, 3)
		z = dual_autodiff.Dual(1*-2, (1*3) +(-2*-2))
		self.assertEqual((x*y).real, z.real)
		self.assertEqual((x*y).dual, z.dual)

	def test_sin(self):
		#x = 
		pass 

if __name__ == '__main__':
	unittest.main() 