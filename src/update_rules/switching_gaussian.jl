@structuredVariationalRule(:node_type     => SwitchingGaussian,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution),
                           :name          => SVBSwitchingGaussianOutNGD)

@structuredVariationalRule(:node_type     => SwitchingGaussian,
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (Message{Gaussian}, Nothing, ProbabilityDistribution),
                          :name           => SVBSwitchingGaussianMGND)

@structuredVariationalRule(:node_type     => SwitchingGaussian,
                        :outbound_type    => Message{Categorical},
                        :inbound_types    => (ProbabilityDistribution, Nothing),
                        :name             => SVBSwitchingGaussianSDN)

@marginalRule(:node_type     => SwitchingGaussian,
              :inbound_types => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
              :name          => MSwitchingGaussianGGD)
