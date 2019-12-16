@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGaussianControlledVarianceOutNGDDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                          :outbound_type => Message{GaussianMeanVariance},
                          :inbound_types => (Message{Gaussian},Nothing,ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                          :name          => SVBGaussianControlledVarianceXGNDDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                        :outbound_type => Message{ExponentialLinearQuadratic},
                        :inbound_types => (ProbabilityDistribution,Nothing,ProbabilityDistribution,ProbabilityDistribution),
                        :name          => SVBGaussianControlledVarianceZDNDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                        :outbound_type => Message{ExponentialLinearQuadratic},
                        :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,Nothing,ProbabilityDistribution),
                        :name          => SVBGaussianControlledVarianceΚDDND)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                       :outbound_type => Message{ExponentialLinearQuadratic},
                       :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,Nothing),
                       :name          => SVBGaussianControlledVarianceΩDDDN)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                          :outbound_type => Message{GaussianMeanVariance},
                          :inbound_types => (Nothing, Message{ExponentialLinearQuadratic}, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                          :name          => SVBGaussianControlledVarianceOutNEDDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                         :outbound_type => Message{GaussianMeanVariance},
                         :inbound_types => (Message{ExponentialLinearQuadratic}, Nothing, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                         :name          => SVBGaussianControlledVarianceMENDDD)

@marginalRule(:node_type     => GaussianControlledVariance,
              :inbound_types => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
              :name          => MGaussianControlledVarianceGGDDD)

@marginalRule(:node_type     => GaussianControlledVariance,
            :inbound_types => (Message{ExponentialLinearQuadratic},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
            :name          => MGaussianControlledVarianceEGDDD)

@marginalRule(:node_type     => GaussianControlledVariance,
            :inbound_types => (Message{Gaussian},Message{ExponentialLinearQuadratic},ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
            :name          => MGaussianControlledVarianceGEDDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                         :outbound_type => Message{GaussianMeanVariance},
                         :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution),
                         :name          => SVBGaussianControlledVarianceOutNGDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                       :outbound_type => Message{GaussianMeanVariance},
                       :inbound_types => (Message{Gaussian},Nothing,ProbabilityDistribution, ProbabilityDistribution),
                       :name          => SVBGaussianControlledVarianceXGNDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                       :outbound_type => Message{GaussianMeanVariance},
                       :inbound_types => (ProbabilityDistribution,Nothing,Message{Gaussian},ProbabilityDistribution),
                       :name          => SVBGaussianControlledVarianceZDGGD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (ProbabilityDistribution,Message{Gaussian},Nothing,ProbabilityDistribution),
                      :name          => SVBGaussianControlledVarianceΚDGGD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                     :outbound_type => Message{ExponentialLinearQuadratic},
                     :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,Nothing),
                     :name          => SVBGaussianControlledVarianceΩDDN)

@marginalRule(:node_type     => GaussianControlledVariance,
           :inbound_types => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution),
           :name          => MGaussianControlledVarianceGGDD)

@marginalRule(:node_type     => GaussianControlledVariance,
            :inbound_types => (ProbabilityDistribution,Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
            :name          => MGaussianControlledVarianceDGGD)
