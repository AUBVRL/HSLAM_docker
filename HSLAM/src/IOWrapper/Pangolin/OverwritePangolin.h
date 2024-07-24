#ifndef __PANGOVERWRITE__
#define __PANGOVERWRITE__

#include <pangolin/display/display_internal.h>
#include <pangolin/gl/glfont.h>

//Overwrite some of pangolin's internal functions.
extern "C" const unsigned char AnonymousPro_ttf[];

namespace pangolin
{

    template <typename T>
    void GuiVarChanged(Var<T> &var)
    {
        VarState::I().FlagVarChanged();
        var.Meta().gui_changed = true;

        for (std::vector<GuiVarChangedCallback>::iterator igvc = VarState::I().gui_var_changed_callbacks.begin(); igvc != VarState::I().gui_var_changed_callbacks.end(); ++igvc)
        {
            if (StartsWith(var.Meta().full_name, igvc->filter))
            {
                igvc->fn(igvc->data, var.Meta().full_name, var.Ref());
            }
        }
    }

    std::mutex new_display_mutex;
    GLfloat Transparent[4] = {0.0f, 0.0f, 0.0f, 0.0f}; //fully transparent color
    GLfloat BlackTransparent[4] = {0.0f, 0.0f, 0.0f, 0.55f}; //fully transparent color
    GLfloat GreenTransparent[4] = {0.0f, 1.0f, 0.0f, 0.5f};
    GLfloat Black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    GLfloat White[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat RedTransparent[4] = {1.0f, 0.0f, 0.0f, 0.8f};
    
    GlFont myfont(AnonymousPro_ttf, 22);
    GlFont largeFont(AnonymousPro_ttf, 32);
    GLfloat *TextColor = White; //{1.0f, 1.0f, 1.0f, 1.0f};

    void glRect(Viewport v)
    {
        GLfloat vs[] = {(float)v.l, (float)v.b,
                        (float)v.l, (float)v.t(),
                        (float)v.r(), (float)v.t(),
                        (float)v.r(), (float)v.b};

        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, vs);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glDisableClientState(GL_VERTEX_ARRAY);
    }

    struct PANGOLIN_EXPORT HandlerResize : Handler
    {

        void Mouse(View &In_, MouseButton button, int x, int y, bool pressed, int button_state)
        {
            float TopBound = In_.top.p;

            if (button == MouseWheelDown) //MouseButtonRight
                TopBound = std::max(0.1, TopBound - 0.01);

            if (button == MouseWheelUp) //MouseButtonLeft
                TopBound = std::min(1.0, TopBound + 0.01);

            In_.SetBounds(In_.bottom, TopBound, In_.left, In_.right);
        }

        void MouseMotion(View &, int x, int y, int button_state)
        {
        }
    };

    struct NewCheckBox : public Checkbox
    {
        NewCheckBox(std::string title, VarValueGeneric &tv) : Checkbox(title, tv) {}

        void Render()
        {
            const bool val = var->Get();

            if (val)
            {
                glColor4fv(GreenTransparent); //when activated
                glRect(vcb);
            }
            glColor4fv(Black);
            gltext = myfont.Text(var->Meta().friendly);
            gltext.DrawWindow(raster[0], raster[1]);
            // DrawShadowRect(vcb, val);
        }
    };

    struct NewSlider : public Slider
    {
        NewSlider(std::string title, VarValueGeneric &tv) : Slider(title, tv) {}
        void Render()
        {
            const double val = var->Get();

            if (var->Meta().range[0] != var->Meta().range[1])
            {
                double rval = val;
                if (logscale)
                {
                    rval = log(val);
                }
                glColor4fv(BlackTransparent);
                glRect(v);
                glColor4fv(GreenTransparent);
                const double norm_val = std::max(0.0, std::min(1.0, (rval - var->Meta().range[0]) / (var->Meta().range[1] - var->Meta().range[0])));
                glRect(Viewport(v.l, v.b, (int)(v.w * norm_val), v.h));
                // DrawShadowRect(v);
                glColor4fv(TextColor); //colour_tx
                gltext = myfont.Text(var->Meta().friendly);
                gltext.DrawWindow(raster[0], raster[1]);
                std::ostringstream oss;
                oss << std::setprecision(5) << val;
                std::string str = oss.str();
                GlText glval = myfont.Text(str);
                const float l = glval.Width() + 2.0f;
                glval.DrawWindow(v.l + v.w - l, raster[1]);
            }
            else
            {
                glColor4fv(Black); //RedTransparent
                //if (gltext.Text() != var->Meta().friendly)
                {
                    gltext = largeFont.Text(var->Meta().friendly);
                }
                gltext.DrawWindow(raster[0], raster[1]);

                std::ostringstream oss;
                oss << std::setprecision(5) << val;
                std::string str = oss.str();
                GlText glval = largeFont.Text(str);
                const float l = glval.Width() + 2.0f;
                glval.DrawWindow(v.l + v.w - l, raster[1]);
            }
        }
    };

    struct NewButton : public Button
    { //KNOWN BUG: this button was created for the recorder URI, when the window is resized the recorder stops working.
        //Since there is no feedback from the recorder the button assumes is still on.
        bool IsOn = false;
        GlText textIfOn;
        GlText textIfOff;
        bool ChangeableName = false;
        NewButton(std::string title, VarValueGeneric &tv) : Button(title, tv)
        {
            if (title.find('!') != std::string::npos) //ex: Record Video!Stop Recording
            {
                gltext = textIfOff = myfont.Text(title.substr(0, title.find('!'))); //Record Video
                textIfOn = myfont.Text(title.substr(title.find('!') + 1));          //Stop Recording
                ChangeableName = true;
            }
            else
                gltext = textIfOff = textIfOn = myfont.Text(title);
        }

        void Render()
        {
            glColor4fv(BlackTransparent);
            glRect(v);
            glColor4fv(TextColor);
            if (ChangeableName)
                if (IsOn)
                    textIfOn.DrawWindow(raster[0], raster[1] - down);
                else
                    textIfOff.DrawWindow(raster[0], raster[1] - down);
            else
                gltext.DrawWindow(raster[0], raster[1] - down);
            // DrawShadowRect(v, down);
        }

        void Mouse(View &, MouseButton button, int /*x*/, int /*y*/, bool pressed, int /*mouse_state*/)
        {
            if (button == MouseButtonLeft)
            {
                down = pressed;
                if (!pressed)
                {
                    IsOn = !IsOn;
                    var->Set(!var->Get());
                    GuiVarChanged(*this);
                }
            }
        }
    };

    struct NewPanel : public Panel
    {
        void Render()
        {
#ifndef HAVE_GLES
            glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_SCISSOR_BIT | GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT | GL_TRANSFORM_BIT);
#endif
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            DisplayBase().ActivatePixelOrthographic();
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_LIGHTING);
            glDisable(GL_SCISSOR_TEST);
            glDisable(GL_LINE_SMOOTH);
            glDisable(GL_COLOR_MATERIAL);
            glLineWidth(1.0);

            glColor4fv(Transparent);

            RenderChildren();

#ifndef HAVE_GLES
            glPopAttrib();
#else
            glEnable(GL_LINE_SMOOTH);
            glEnable(GL_DEPTH_TEST);
#endif
        }

        static void AddNewVariable(void *data, const std::string &name, VarValueGeneric &var, bool /*brand_new*/)
        {
            NewPanel *thisptr = (NewPanel *)data;

            const std::string &title = var.Meta().friendly;

            new_display_mutex.lock();

            ViewMap::iterator pnl = GetCurrentContext()->named_managed_views.find(name);

            if (pnl == GetCurrentContext()->named_managed_views.end())
            {
                View *nv = NULL;
                if (!strcmp(var.TypeId(), typeid(bool).name()))
                {
                    nv = (var.Meta().flags & META_FLAG_TOGGLE) ? (View *)new NewCheckBox(title, var) : (View *)new NewButton(title, var);
                }
                else if (!strcmp(var.TypeId(), typeid(double).name()) ||
                         !strcmp(var.TypeId(), typeid(float).name()) ||
                         !strcmp(var.TypeId(), typeid(int).name()) ||
                         !strcmp(var.TypeId(), typeid(unsigned int).name()))
                {
                    nv = new NewSlider(title, var);
                }
                else if (!strcmp(var.TypeId(), typeid(std::function<void(void)>).name()))
                {
                    nv = (View *)new FunctionButton(title, var);
                }
                else
                {
                    nv = new TextInput(title, var);
                }
                if (nv)
                {
                    GetCurrentContext()->named_managed_views[name] = nv;
                    thisptr->views.push_back(nv);
                    thisptr->ResizeChildren();
                }
            }

            new_display_mutex.unlock();
        }

        NewPanel(const std::string &auto_register_var_prefix) : Panel()
        {
            pangolin::RegisterNewVarCallback(&AddNewVariable, (void *)this, auto_register_var_prefix);
            pangolin::ProcessHistoricCallbacks(&AddNewVariable, (void *)this, auto_register_var_prefix);
        }
    };

    PANGOLIN_EXPORT View &CreateNewPanel(const std::string &name)
    {
        if (GetCurrentContext()->named_managed_views.find(name) != GetCurrentContext()->named_managed_views.end())
        {
            throw std::runtime_error("Panel already registered with this name.");
        }
        NewPanel *p = new NewPanel(name);
        GetCurrentContext()->named_managed_views[name] = p;
        GetCurrentContext()->base.views.push_back(p);
        return *p;
    }
} // namespace pangolin

#endif
