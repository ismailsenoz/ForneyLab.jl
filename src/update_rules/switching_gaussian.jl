@structuredVariationalRule(:node_type     => SwitchingGaussian,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBSwitchingGaussianOutNGDDD)

@structuredVariationalRule(:node_type     => SwitchingGaussian,
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (Message{Gaussian}, Nothing, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                          :name           => SVBSwitchingGaussianMGNDDD)

@structuredVariationalRule(:node_type     => SwitchingGaussian,
                        :outbound_type    => Message{Categorical},
                        :inbound_types    => (ProbabilityDistribution, Nothing, ProbabilityDistribution,ProbabilityDistribution),
                        :name             => SVBSwitchingGaussianSDNDD)

@marginalRule(:node_type     => SwitchingGaussian,
              :inbound_types => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
              :name          => MSwitchingGaussianGGDDD)
