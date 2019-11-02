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

@marginalRule(:node_type     => GaussianControlledVariance,
              :inbound_types => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
              :name          => MGaussianControlledVarianceGGDDD)

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
                       :name          => SVBGaussianControlledVarianceZDNGD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (ProbabilityDistribution,Message{Gaussian},Nothing,ProbabilityDistribution),
                      :name          => SVBGaussianControlledVarianceΚDGND)

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
