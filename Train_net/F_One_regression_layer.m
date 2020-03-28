classdef F_One_regression_layer < nnet.layer.RegressionLayer
        %Regression layer for cyclic output, angle between 0 to 359.
        %Squared loss function.
    properties
        % (Optional) Layer properties.
    end
 
    methods
        function layer = F_One_regression_layer(name)           
            layer.Name = name;
            layer.Description = 'Loss for Cyclical tags and 2D input';
        end

        function loss = forwardLoss(~, Y, T)
            % Return the squared loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training 
            %
            % Output:
            %         loss  - Loss between Y and T

            % l is the vector of distances between Y to T
            l = mod(abs(wrapTo360(Y) - T), 360); %assigen 360 to 0;
            
            % When talking about distance between angles we can have two solutions.
            % We will take  the smaller one. For example the distance
            % between 180 to 60 can be (180-60) or (60-180), we would like
            % to take (180-60).

            ind_vec = l > 180;
            l = l.*(~ind_vec) + (360 - l).*ind_vec;
            %calculate loss
            loss = sum(l.^2) / length(l);
        end
        
        function dLdY = backwardLoss(~, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training 
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y        

                        
            % calculate the direction of the derivative according to the (T - Y)
            % tree.
            l_dircetion = mod(T - wrapTo360(Y), 360); %assigen 360 to 0;
            zero_ind_vec = l_dircetion > 0;
            ind_180_vec = abs(l_dircetion) > 180;
            l_dircetion = (l_dircetion.*((~ind_180_vec)) + l_dircetion.*((-1)*(ind_180_vec))).*zero_ind_vec...
                + (l_dircetion.*((ind_180_vec)) + l_dircetion.*((-1)*(~ind_180_vec))).*(~zero_ind_vec);
            
            %calculate the abs value of the derivative according, the
            %distance T - Y, as before, we will take the small distance.
            l_value = mod(abs(wrapTo360(Y) - T), 360); %assigen 360 to 0;
            ind_vec = l_value > 180;
            l_value = l_value.*(~ind_vec) + (360 - l_value).*ind_vec;
            
            %combaine the direction and the value
            zero_ind_vec = l_dircetion > 0;
            l = l_value.*((-1)*zero_ind_vec) + l_value.*(~zero_ind_vec);
            dLdY = l*2 / length(l);

        end
    end
end

