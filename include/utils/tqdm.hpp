#pragma once

#include <string>
#include <iostream>

namespace utils
{

  class TQDM
  {
  private:
    size_t N_{0};
    size_t curr_{0};
    std::string name_;
    std::string bar_;
    size_t TQDM_WIDTH{50};

  public:
    explicit TQDM(size_t N, const std::string &name = "") : N_(N), name_(name)
    {
      if (N < TQDM_WIDTH)
        TQDM_WIDTH = N;
      bar_rebuild();
      curr_ = 0;
      std::cout << name_ << bar_;
    }

    void reset()
    {
      bar_rebuild();
      curr_ = 0;
      update_bar();
    }

    void next()
    {
      curr_++;
      if (curr_ > N_)
        return;
      size_t idx = curr_ * TQDM_WIDTH / N_;
      if (idx)
        bar_[idx] = '=';
      update_bar();
    }

    ~TQDM()
    {
      std::cout << std::endl;
    }

  private:
    inline void bar_rebuild()
    {
      bar_ = std::string(TQDM_WIDTH + 2, ' ');
      bar_[0] = '[';
      bar_[TQDM_WIDTH + 1] = ']';
    }

    inline void update_bar() const
    {
      for (size_t i = 0; i < TQDM_WIDTH + 2; i++)
      {
        std::cout << '\b';
      }
      std::cout << bar_;
      flush(std::cout);
    }
  };

}