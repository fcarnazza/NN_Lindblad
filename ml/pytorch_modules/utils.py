import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

def kronecker_product(mat1, mat2):
    out_mat = torch.ger(mat1.view(-1), mat2.view(-1))
    out_mat = out_mat.reshape(*(mat1.size() + mat2.size())).permute([0, 2, 1, 3])
    out_mat = out_mat.reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))
    return out_mat 

def main():
    from torch.autograd import gradcheck
    k = torch.randn(20, 10).double()
    # Create a positive definite matrix
    pd_mat = (k.t().matmul(k)).requires_grad_()
    test = gradcheck(sqrtm, (pd_mat,))
    print(test)


def one_negative(x):
        it = False
        if any(t < 0 for t in x):
                it = True
        return it

def simplexion(x):
        I =[]
        while one_negative(x): 
                s_x_i = x.sum()-x[I].sum()
                for i in range(len(x)):     
                        if not i in I:
                                x[i] = x[i]+(1-s_x_i)/(len(x) - len(I)) 
                                if x[i]<0:
                                          I = list(set(I)|set([i]))             
                        else:
                                x[i] = 0
        
        return x



if __name__ == '__main__':
    main()
