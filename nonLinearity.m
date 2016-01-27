function value = nonLinearity(func, x, type)
    if nargin < 2
      error('input_example :  func, and data parameters are required')
    end
    if nargin < 3
      type = 'regular';
    end
    
    switch type
        case 'derivative'
            if(strcmp(func, 'sigmoid') == 1)
                value = x.*(1-x);
            elseif(strcmp(func, 'hyperbolic') == 1)
                value = sech(x).*sech(x);
            elseif(strcmp(func, 'rectifier') == 1)
                value = 1./(1 + exp(-x));
            else
                error('Incorrect parameter. FUNC = sigmoid, hyperbolic, or rectifier')
            end
        case 'regular'
            if(strcmp(func, 'sigmoid') == 1)
                value = 1./(1+exp(-x));
            elseif(strcmp(func, 'hyperbolic') == 1)
                value = tanh(x);
            elseif(strcmp(func, 'rectifier') == 1)
                value = log(1 + exp(x));
            else
                error('Incorrect parameter. FUNC = sigmoid, hyperbolic, or rectifier')
            end
        otherwise
            error('Incorrect parameter. TYPE = derivative or regular')
    end
end

