function c = cifar_10_bayes_classify2(f,mu,sigma,p)
    for cl = 1:length(p)
        posteriori(cl,:)=mvnpdf(f,mu(cl,:),sigma)*p(cl);
    end
    [~, c] = max(posteriori);
end


