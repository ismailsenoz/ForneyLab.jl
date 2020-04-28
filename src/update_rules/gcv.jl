@structuredVariationalRule(:node_type      => GCV{Cubature},
                           :outbound_type  => Message{GaussianMeanVariance},
                           :inbound_types  => (Nothing, Message{Gaussian},ProbabilityDistribution),
                           :name           => SVBGCVCubatureOutNGD)

@structuredVariationalRule(:node_type      => GCV{Cubature},
                           :outbound_type  => Message{GaussianMeanVariance},
                           :inbound_types  => (Message{Gaussian},Nothing,ProbabilityDistribution),
                           :name           => SVBGCVCubatureMGND)

@structuredVariationalRule(:node_type      => GCV{Cubature},
                           :outbound_type  => Message{Function},
                           :inbound_types  => (ProbabilityDistribution,Nothing),
                           :name           => SVBGCVCubatureZDN)

@marginalRule(:node_type      => GCV{Cubature},
              :inbound_types  => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
              :name           => MGCVCubatureMGGD)


@structuredVariationalRule(:node_type      => GCV{Laplace},
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (Nothing, Message{Gaussian},ProbabilityDistribution),
                          :name           => SVBGCVLaplaceOutNGD)

@structuredVariationalRule(:node_type      => GCV{Laplace},
                          :outbound_type  => Message{GaussianMeanVariance},
                          :inbound_types  => (Message{Gaussian},Nothing,ProbabilityDistribution),
                          :name           => SVBGCVLaplaceMGND)

@structuredVariationalRule(:node_type      => GCV{Laplace},
                          :outbound_type  => Message{Function},
                          :inbound_types  => (ProbabilityDistribution,Nothing),
                          :name           => SVBGCVLaplaceZDN)

@marginalRule(:node_type      => GCV{Laplace},
             :inbound_types  => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution),
             :name           => MGCVLaplaceMGGD)
