% CODED BY: PUNEET DHEER (RF) 
% UPDATE ON: 11-MAY-2017
%
% Interaction_Information_SYNERGY AND REDUNDANCY
%
% INPUT:
% rowwise_signal = rowwise signal (each row is channel)
% Ws = window size in sample point
% Lp = Shift the window by some sample point
%
% OUTPUT:
% G_L = Synergy'1' and Redundancy'0' binary values to check whether Synergy and Redundancy is happening (IN MUTUALLY EXCLUSIVE MANNER)
% ENN = All entropy values from single to all pairs in row wise manner
% I_I = main output of this code (INTERACTION Information)
% sENN = signed entropy just for checking purpose
%
% REFERENCES:
% [1] https://link.springer.com/article/10.1007/BF02289159
% [2] https://arxiv.org/abs/cs/0308002
% [3] https://en.wikipedia.org/wiki/Interaction_information
% [4] https://www.sciencedirect.com/science/article/pii/S0019995880904787
%
% Interaction information can either be positive or negative

function [G_L,ENN,I_I,sENN]= Interaction_Information_permut_MV(rowwise_signal,Ws,Lp)
tic
Lw=1;
z=Ws;
jjj=0;
tic


xx=rowwise_signal;
No_of_variable=size(xx,1);
c=cell(1,No_of_variable); 

for j=1:No_of_variable
   c{1,j} = combnk(1:No_of_variable,j);% all possible pairs
%    tnum=tnum+size(c{1,j},1);
end

kk=1;
jj=1;
total_num=0;
for iii=1:No_of_variable %no of channels
    cellsize=size(c{1,iii},1);
    
    for j=1:cellsize %total rows in each iii cell
        content=c{1,iii}(j,:);
        total_num=total_num+1;
        Ws=Ws;   
        Lp=Lp; 
        Lw=1;
        z=Ws;
        
        if length(content)==1
            fprintf('%d_%d/%d_pair/%d_pair\n',j,cellsize,iii,No_of_variable);
            for i=1:ceil((length(xx)-Ws+1)/Lp)
                X=(xx(content,Lw:z));
                
                Lw=Lw+Lp;
                z=z+Lp;
                
                OPxI = (permut(X));   %feature in signal x
                OP{j,i}=OPxI; %to store the feature in window wise manner
                unixI=unique(OPxI); %extract unique value
                outxI = [unixI;histc(OPxI,unixI)]; %number of occurence of each unique value
                Probx=sum(outxI(2,:));
                Probx=outxI(2,:)/sum(Probx);
                
                ENx(kk,i)=-sum(Probx .* log(Probx)); %ENTROPY
                
                
            end
            kk=kk+1;
            
        else
            
            fprintf('%d_%d/%d_pair/%d_pair\n',j,cellsize,iii,No_of_variable);
            cc=content;
            for i=1:length(OP) %upto number of windows
                Hxy = OP((cc),i);%row wise channels
                Hxy = squeeze(Hxy);
                Hxy = cell2mat(Hxy);
                [unique_rows_xy,~,ind]=unique(Hxy','rows');
               
                Hxycounts = histc(ind,unique(ind));
                % counts = histc(ind,1:max(ind));
                Probxy=sum(Hxycounts);  %total counts
                Probxy=Hxycounts/sum(Probxy);
                ENxy(jj,i)=-sum(Probxy .* log(Probxy)); %ENTROPY OF XY
            end
            jj=jj+1;
        end
        
    end

end

ENN=[ENx;ENxy];
dbstop if error
total_num=0;
for i=1:No_of_variable %no_of_channel
    cellsize=size(c{1,i},1); %total content in each cell
    for j=1:cellsize
        content=c{1,i}(j,:);
        lnth=length(content);
        total_num=total_num+1; %total combination
    end
end
    
df=1;
for i=1:No_of_variable %no_of_channel
    cellsize=size(c{1,i},1);%total content in each cell
    for j=1:cellsize
        content=c{1,i}(j,:);
        lnth(df)=length(content); %length of content in each cell
        sign(df)=((-1).^(total_num-lnth(df))); %Jakulin & Bratko 2008
        df=df+1;
    end
end     

for i=1:size(ENN,2) %window wise
    
    sENN(:,i)=ENN(:,i).*sign';
    dbstop if error
    
    cmnMI(:,i)=-(sum(sENN(:,i)));
    dbstop if error
    
    if cmnMI(:,i)<0 
        I_I(i)=abs(cmnMI(:,i));
        dbstop if error
        disp('REDUNDANCY LOSS and Negative interaction information')
        G_L(i)=0; %0 indicates redundancy of information (REDUNDANCY) 
        % between X and Y (provide the same information) when Z is taken into account
        % More simply said, X and Y does not give additional information
        % for the prediction of target Z
    else
        I_I(i)=abs((cmnMI(:,i)));
        dbstop if error
        disp('SYNERGY GAIN and Positive interaction information')
        G_L(i)=1; %1 indicates gain in information (SYNERGY)
        % More simply said, X and Y together enhances the prediction of
        % target Z
    end

end
    %NOTE: x,y,z are three variables
    %-ve II means information is decreased i.e., I(X;Y)>=I(X,Y|Z) so
    %MIxyz <= MIxy
    %+ve II means information is increased i.e., I(X;Y)<=I(X,Y|Z) so
    %MIxyz >= MIxy
    %MIxyz==MIxy iff I(X,Y|Z)=0

toc
